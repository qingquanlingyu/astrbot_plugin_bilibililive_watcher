from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

PcmCallback = Callable[[bytes], Awaitable[None]]
StderrCallback = Callable[[str], Awaitable[None] | None]

DEFAULT_AUDIO_ORIGIN = "https://live.bilibili.com"
DEFAULT_STDERR_BUFFER_LINES = 20


@dataclass(frozen=True, slots=True)
class AudioRequestOptions:
    room_id: int
    user_agent: str
    origin: str = DEFAULT_AUDIO_ORIGIN
    referer: str = ""
    cookie: str = ""

    @classmethod
    def for_room(
        cls,
        *,
        room_id: int,
        user_agent: str,
        cookie: str = "",
        origin: str = DEFAULT_AUDIO_ORIGIN,
    ) -> "AudioRequestOptions":
        return cls(
            room_id=room_id,
            user_agent=user_agent,
            origin=origin,
            referer=f"{origin.rstrip('/')}/{room_id}",
            cookie=cookie,
        )

    def build_header_blob(self) -> str:
        lines = []
        if self.referer:
            lines.append(f"Referer: {self.referer}")
        if self.origin:
            lines.append(f"Origin: {self.origin}")
        if self.user_agent:
            lines.append(f"User-Agent: {self.user_agent}")
        if self.cookie:
            lines.append(f"Cookie: {self.cookie}")
        if not lines:
            return ""
        return "\r\n".join(lines) + "\r\n"


class AudioCaptureWorker:
    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        sample_rate: int = 16000,
        chunk_ms: int = 100,
        read_timeout_seconds: float = 15.0,
    ):
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.sample_rate = max(8000, int(sample_rate or 16000))
        self.chunk_ms = max(20, int(chunk_ms or 100))
        self.read_timeout_seconds = max(0.0, float(read_timeout_seconds or 0.0))
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_lines: list[str] = []

    def build_ffmpeg_command(
        self,
        stream_url: str,
        *,
        request_options: AudioRequestOptions | None = None,
    ) -> list[str]:
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]
        if request_options is not None:
            header_blob = request_options.build_header_blob()
            if header_blob:
                cmd.extend(["-headers", header_blob])
        cmd.extend(
            [
                "-i",
                stream_url,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-f",
                "s16le",
                "pipe:1",
            ]
        )
        return cmd

    async def run(
        self,
        stream_url: str,
        on_pcm: PcmCallback,
        *,
        request_options: AudioRequestOptions | None = None,
        on_stderr: StderrCallback | None = None,
    ):
        bytes_per_second = self.sample_rate * 2  # s16le mono
        chunk_bytes = max(320, int(bytes_per_second * self.chunk_ms / 1000))
        cmd = self.build_ffmpeg_command(stream_url, request_options=request_options)
        self._stderr_lines = []
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stderr_task = asyncio.create_task(
            self._consume_stderr(self._proc.stderr, on_stderr),
            name="ffmpeg-stderr-reader",
        )
        try:
            while True:
                read_coro = self._proc.stdout.read(chunk_bytes)
                if self.read_timeout_seconds > 0:
                    try:
                        chunk = await asyncio.wait_for(
                            read_coro,
                            timeout=self.read_timeout_seconds,
                        )
                    except asyncio.TimeoutError as e:
                        raise RuntimeError(
                            "audio stream stalled: no PCM for "
                            f"{self.read_timeout_seconds:.1f}s{self._format_stderr_suffix()}"
                        ) from e
                else:
                    chunk = await read_coro
                if not chunk:
                    break
                await on_pcm(chunk)
            returncode = await self._proc.wait()
            await self._drain_stderr_task()
            if returncode not in (0, None):
                raise RuntimeError(
                    f"ffmpeg exited with code {returncode}{self._format_stderr_suffix()}"
                )
        finally:
            await self.stop()

    async def stop(self):
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        await self._drain_stderr_task()
        self._proc = None

    async def _consume_stderr(
        self,
        stream: asyncio.StreamReader,
        on_stderr: StderrCallback | None,
    ) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            self._stderr_lines.append(line)
            if len(self._stderr_lines) > DEFAULT_STDERR_BUFFER_LINES:
                self._stderr_lines = self._stderr_lines[-DEFAULT_STDERR_BUFFER_LINES:]
            if on_stderr is None:
                continue
            result = on_stderr(line)
            if result is not None:
                await result

    def _format_stderr_suffix(self) -> str:
        if not self._stderr_lines:
            return ""
        return f" | stderr: {' || '.join(self._stderr_lines[-3:])}"

    async def _drain_stderr_task(self) -> None:
        if self._stderr_task is None:
            return
        await asyncio.gather(self._stderr_task, return_exceptions=True)
        self._stderr_task = None

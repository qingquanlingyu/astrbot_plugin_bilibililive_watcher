from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

PcmCallback = Callable[[bytes], Awaitable[None]]

DEFAULT_AUDIO_ORIGIN = "https://live.bilibili.com"


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
    ):
        bytes_per_second = self.sample_rate * 2  # s16le mono
        chunk_bytes = max(320, int(bytes_per_second * self.chunk_ms / 1000))
        cmd = self.build_ffmpeg_command(stream_url, request_options=request_options)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        assert self._proc.stdout is not None
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
                            f"audio stream stalled: no PCM for {self.read_timeout_seconds:.1f}s"
                        ) from e
                else:
                    chunk = await read_coro
                if not chunk:
                    break
                await on_pcm(chunk)
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
        self._proc = None

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Awaitable, Callable

if __package__:
    from .audio_pipe import AudioRequestOptions
    from .bili_http import DEFAULT_UA, BiliHttpClient
    from .recording_manifest import (
        RecordingSegment,
        SessionLayout,
        append_segment_index,
        build_session_layout,
        init_session_index,
        update_session_index,
    )
else:  # pragma: no cover
    from audio_pipe import AudioRequestOptions
    from bili_http import DEFAULT_UA, BiliHttpClient
    from recording_manifest import (
        RecordingSegment,
        SessionLayout,
        append_segment_index,
        build_session_layout,
        init_session_index,
        update_session_index,
    )

SegmentCallback = Callable[[RecordingSegment], Awaitable[None] | None]


class LiveRecorderRuntime:
    def __init__(
        self,
        *,
        http_client: BiliHttpClient,
        room_id: int,
        real_room_id: int,
        output_root: str,
        ffmpeg_path: str,
        segment_duration_seconds: int,
        output_container: str,
        pull_protocol: str,
        api_preference: str,
        cookie: str,
        audio_http_headers_enabled: bool,
        max_session_hours: int,
        session_extra: dict[str, object] | None = None,
        on_segment: SegmentCallback | None = None,
    ):
        self.http_client = http_client
        self.room_id = int(room_id)
        self.real_room_id = int(real_room_id)
        self.ffmpeg_path = str(ffmpeg_path or "ffmpeg").strip() or "ffmpeg"
        self.segment_duration_seconds = max(30, int(segment_duration_seconds or 300))
        self.output_container = str(output_container or "mkv").strip() or "mkv"
        self.pull_protocol = str(pull_protocol or "http_flv").strip() or "http_flv"
        self.api_preference = str(api_preference or "getRoomPlayInfo").strip() or "getRoomPlayInfo"
        self.cookie = str(cookie or "").strip()
        self.audio_http_headers_enabled = bool(audio_http_headers_enabled)
        self.max_session_hours = max(1, int(max_session_hours or 12))
        self.session_extra = dict(session_extra or {})
        self.on_segment = on_segment
        self.started_at = time.time()
        self.layout = build_session_layout(output_root, self.real_room_id or self.room_id, self.started_at)
        self.session_id = self.layout.root.name
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._segment_count = 0
        self._last_error = ""
        self._last_segment_id = ""
        self._last_probe_url = ""
        init_session_index(
            self.layout.root,
            room_id=self.room_id,
            real_room_id=self.real_room_id,
            started_at=self.started_at,
            session_id=self.session_id,
            notes=[
                "recording runtime is isolated from danmaku/asr runtime",
                "history danmaku timestamps are observational, not precise clip boundaries",
            ],
            extra=self.session_extra,
        )

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._task = asyncio.create_task(self.run_loop(), name=f"bili-recording-{self.real_room_id}")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        update_session_index(self.layout.root, ended_at=time.time())

    async def run_loop(self) -> None:
        deadline = self.started_at + (self.max_session_hours * 3600)
        backoff = 1.0
        while not self._stop_event.is_set() and time.time() < deadline:
            segment_index = self._segment_count + 1
            segment_id = f"segment-{segment_index:04d}"
            segment_path = self._segment_path(segment_index)
            try:
                urls = await self.http_client.get_room_play_urls(
                    room_id=self.real_room_id,
                    cookie=self.cookie,
                    pull_protocol=self.pull_protocol,
                    api_preference=self.api_preference,
                )
                if not urls:
                    raise RuntimeError("no play url available")
                stream_url = urls[0]
                self._last_probe_url = stream_url
                record = await self._record_single_segment(
                    segment_id=segment_id,
                    stream_url=stream_url,
                    output_path=segment_path,
                    duration_seconds=float(self.segment_duration_seconds),
                )
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                record = RecordingSegment(
                    segment_id=segment_id,
                    file_path=str(segment_path),
                    wall_ts_start=time.time(),
                    wall_ts_end=time.time(),
                    duration_seconds=0.0,
                    stream_url_fingerprint="",
                    ok=False,
                    returncode=-1,
                    error=str(exc),
                )
                await asyncio.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)
            self._segment_count += 1
            self._last_segment_id = record.segment_id
            append_segment_index(self.layout.root, record)
            if self.on_segment is not None:
                result = self.on_segment(record)
                if result is not None:
                    await result
        update_session_index(self.layout.root, ended_at=time.time())

    def status_snapshot(self) -> dict[str, object]:
        return {
            "running": self.running,
            "session_id": self.session_id,
            "session_root": str(self.layout.root),
            "segment_count": self._segment_count,
            "last_segment_id": self._last_segment_id,
            "last_error": self._last_error,
        }

    def _segment_path(self, index: int) -> Path:
        return self.layout.segments_dir / f"segment-{index:04d}.{self.output_container}"

    async def _record_single_segment(
        self,
        *,
        segment_id: str,
        stream_url: str,
        output_path: Path,
        duration_seconds: float,
    ) -> RecordingSegment:
        header_blob = ""
        if self.audio_http_headers_enabled:
            header_blob = AudioRequestOptions.for_room(
                room_id=self.real_room_id,
                user_agent=DEFAULT_UA,
                cookie=self.cookie,
            ).build_header_blob()
        cmd = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error"]
        if header_blob:
            cmd.extend(["-headers", header_blob])
        cmd.extend(
            [
                "-y",
                "-i",
                stream_url,
                "-t",
                f"{duration_seconds:.3f}",
                "-map",
                "0",
                "-c",
                "copy",
                str(output_path),
            ]
        )
        wall_ts_start = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_lines: list[str] = []
        assert proc.stderr is not None
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            stderr_lines.append(line)
            if len(stderr_lines) > 5:
                stderr_lines = stderr_lines[-5:]
        returncode = await proc.wait()
        wall_ts_end = time.time()
        error = " | ".join(stderr_lines[-3:])
        return RecordingSegment(
            segment_id=segment_id,
            file_path=str(output_path),
            wall_ts_start=wall_ts_start,
            wall_ts_end=wall_ts_end,
            duration_seconds=max(0.0, wall_ts_end - wall_ts_start),
            stream_url_fingerprint=self._fingerprint_stream_url(stream_url),
            ok=returncode == 0 and output_path.exists(),
            returncode=returncode,
            error=error,
        )

    def _fingerprint_stream_url(self, stream_url: str) -> str:
        text = str(stream_url or "").strip()
        if not text:
            return ""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
import types
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import aiohttp


def _install_fake_astrbot_modules():
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")

    class _Logger:
        def info(self, *args, **kwargs):
            if args:
                print(*args, flush=True)

        def warning(self, *args, **kwargs):
            if args:
                print(*args, flush=True)

    api.logger = _Logger()
    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api


_install_fake_astrbot_modules()

try:  # pragma: no cover
    from .asr_sherpa import ASRDebugEvent, build_asr_worker_or_none
    from .audio_pipe import AudioRequestOptions
    from .bili_http import BiliHttpClient, DEFAULT_UA
    from .bili_ws import DanmakuRealtimeClient
    from .models import ASRSegment, DanmakuItem
except ImportError:  # pragma: no cover
    from asr_sherpa import ASRDebugEvent, build_asr_worker_or_none
    from audio_pipe import AudioRequestOptions
    from bili_http import BiliHttpClient, DEFAULT_UA
    from bili_ws import DanmakuRealtimeClient
    from models import ASRSegment, DanmakuItem


DEFAULT_ROOM_ID = 27004785
DEFAULT_COOKIE_FILE = "~/.bilibili-cookie.json"
DEFAULT_PLUGIN_CONFIG_FILE = "/mnt/ssd/qq/astrbot/data/config/astrbot_plugin_bilibililive_watcher_config.json"
DEFAULT_OUTPUT_ROOT = "/mnt/ssd/bilibili"
DEFAULT_ASR_MODEL_DIR = (
    "./models/sherpa/rknn/"
    "sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
DEFAULT_ASR_VAD_MODEL_PATH = "./models/vad/silero_vad.onnx"


@dataclass(frozen=True, slots=True)
class SessionLayout:
    root: Path
    segments_dir: Path
    timeline_dir: Path
    clips_dir: Path
    reports_dir: Path


@dataclass(slots=True)
class SegmentRecord:
    segment_id: str
    file_path: str
    wall_ts_start: float
    wall_ts_end: float
    duration_seconds: float
    stream_url: str
    ok: bool
    returncode: int
    error: str = ""


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _build_session_id(started_at: float) -> str:
    return time.strftime("session-%Y%m%d-%H%M%S", time.localtime(started_at))


def build_session_layout(output_root: str | Path, room_id: int, started_at: float) -> SessionLayout:
    root = Path(output_root).expanduser().resolve() / "recordings" / str(room_id) / _build_session_id(started_at)
    segments_dir = root / "segments"
    timeline_dir = root / "timeline"
    clips_dir = root / "clips"
    reports_dir = root / "reports"
    for path in (segments_dir, timeline_dir, clips_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return SessionLayout(
        root=root,
        segments_dir=segments_dir,
        timeline_dir=timeline_dir,
        clips_dir=clips_dir,
        reports_dir=reports_dir,
    )


def _load_cookie_from_plugin_config(plugin_config_file: str) -> str:
    path = Path(plugin_config_file).expanduser()
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return ""
    user_auth = data.get("user_auth", {}) or {}
    manual_cookie = str(user_auth.get("bili_cookie", "") or "").strip()
    if manual_cookie:
        return manual_cookie
    return str(user_auth.get("bili_login_cookie", "") or "").strip()


def _load_cookie(
    raw_cookie: str,
    cookie_file: str,
    *,
    plugin_config_file: str = "",
    cookie_from_plugin_config: bool = False,
) -> str:
    if raw_cookie.strip():
        return raw_cookie.strip()
    if cookie_from_plugin_config:
        plugin_cookie = _load_cookie_from_plugin_config(plugin_config_file)
        if plugin_cookie:
            return plugin_cookie
    path = Path(cookie_file).expanduser()
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("cookie", "") or "").strip()
    except Exception:
        return ""


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def pick_segment_for_clip(
    segments: list[dict],
    clip_start_wall_ts: float,
    clip_duration_seconds: float,
) -> dict:
    clip_end_wall_ts = clip_start_wall_ts + clip_duration_seconds
    for segment in segments:
        start = float(segment.get("wall_ts_start", 0.0) or 0.0)
        end = float(segment.get("wall_ts_end", 0.0) or 0.0)
        if start <= clip_start_wall_ts and end >= clip_end_wall_ts:
            return segment
    raise ValueError(
        "clip window is not fully contained in a single recorded segment; "
        "use a shorter duration or a different segment"
    )


def _format_srt_timestamp(raw_seconds: float) -> str:
    total_ms = max(0, int(round(raw_seconds * 1000)))
    hours = total_ms // 3600000
    total_ms -= hours * 3600000
    minutes = total_ms // 60000
    total_ms -= minutes * 60000
    seconds = total_ms // 1000
    milliseconds = total_ms - (seconds * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def build_srt_text(
    asr_events: list[dict],
    clip_start_wall_ts: float,
    clip_duration_seconds: float,
) -> str:
    clip_end_wall_ts = clip_start_wall_ts + clip_duration_seconds
    cues: list[str] = []
    idx = 1
    for event in asr_events:
        text = str(event.get("text", "") or "").strip()
        start = float(event.get("wall_ts_start", 0.0) or 0.0)
        end = float(event.get("wall_ts_end", 0.0) or 0.0)
        if not text or end <= clip_start_wall_ts or start >= clip_end_wall_ts:
            continue
        rel_start = max(0.0, start - clip_start_wall_ts)
        rel_end = min(clip_duration_seconds, end - clip_start_wall_ts)
        if rel_end <= rel_start:
            continue
        cues.extend(
            [
                str(idx),
                f"{_format_srt_timestamp(rel_start)} --> {_format_srt_timestamp(rel_end)}",
                text,
                "",
            ]
        )
        idx += 1
    return "\n".join(cues).strip() + ("\n" if cues else "")


def _segment_path(layout: SessionLayout, index: int, container: str) -> Path:
    return layout.segments_dir / f"segment-{index:04d}.{container}"


class LiveSmartClipPipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cookie = _load_cookie(
            args.cookie,
            args.cookie_file,
            plugin_config_file=args.plugin_config_file,
            cookie_from_plugin_config=args.cookie_from_plugin_config,
        )
        self.wbi_cookie = str(args.wbi_cookie or "").strip() or self.cookie
        self._stop_event = asyncio.Event()
        self._session: aiohttp.ClientSession | None = None
        self._http: BiliHttpClient | None = None
        self._ws_client: DanmakuRealtimeClient | None = None
        self._audio_task: asyncio.Task | None = None
        self._history_task: asyncio.Task | None = None
        self._asr_worker = None
        self._layout: SessionLayout | None = None
        self._manifest_path: Path | None = None
        self._manifest: dict[str, object] = {}
        self._timeline_events_path: Path | None = None
        self._asr_jsonl_path: Path | None = None

    async def run(self) -> int:
        if self.args.command == "probe":
            return await self._run_probe()
        if self.args.command == "capture":
            return await self._run_capture()
        if self.args.command == "clip":
            return await self._run_clip()
        raise RuntimeError(f"unsupported command: {self.args.command}")

    async def stop(self):
        self._stop_event.set()

    async def _run_probe(self) -> int:
        async with aiohttp.ClientSession(headers={"User-Agent": DEFAULT_UA}) as session:
            http = BiliHttpClient(session)
            room_id = await self._resolve_real_room_id(http, self.args.room_id)
            meta = await http.get_room_prompt_meta(room_id=room_id, cookie=self.cookie)
            urls = await http.get_room_play_urls(
                room_id=room_id,
                cookie=self.cookie,
                pull_protocol=self.args.audio_pull_protocol,
                api_preference=self.args.audio_pull_api_preference,
            )
            print(f"[{_ts()}] room_id={self.args.room_id} real_room_id={room_id}")
            print(f"[{_ts()}] anchor={meta.anchor_name or '-'} title={meta.room_title or '-'}")
            print(f"[{_ts()}] play_url_count={len(urls)}")
            if urls:
                print(f"[{_ts()}] play_url_sample={urls[0]}")
        return 0

    async def _run_capture(self) -> int:
        started_at = time.time()
        self._layout = build_session_layout(self.args.output_root, self.args.room_id, started_at)
        self._manifest_path = self._layout.root / "session_manifest.json"
        self._timeline_events_path = self._layout.timeline_dir / "events.jsonl"
        self._asr_jsonl_path = self._layout.timeline_dir / "asr.jsonl"

        self._session = aiohttp.ClientSession(headers={"User-Agent": DEFAULT_UA})
        self._http = BiliHttpClient(self._session)
        real_room_id = await self._resolve_real_room_id(self._http, self.args.room_id)

        self._manifest = {
            "room_id": self.args.room_id,
            "real_room_id": real_room_id,
            "started_at": started_at,
            "session_dir": str(self._layout.root),
            "segments": [],
            "notes": [
                "publishing is intentionally excluded from this runner",
                "history danmaku timestamps are observational, not precise clip boundaries",
            ],
            "args": vars(self.args),
        }
        self._flush_manifest()

        print(f"[{_ts()}] capture session={self._layout.root}")
        print(f"[{_ts()}] room_id={self.args.room_id} real_room_id={real_room_id}")

        if not self.args.no_ws:
            await self._start_ws(real_room_id)
        if not self.args.no_history:
            self._history_task = asyncio.create_task(
                self._poll_history_loop(real_room_id),
                name="smartclip-history",
            )
        if self.args.asr:
            await self._start_audio_asr(real_room_id)

        try:
            await self._capture_segments_loop(real_room_id)
        finally:
            await self._shutdown_capture()

        return 0

    async def _run_clip(self) -> int:
        session_dir = Path(self.args.session_dir).expanduser().resolve()
        manifest_path = session_dir / "session_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"session manifest missing: {manifest_path}")
        manifest = load_json(manifest_path)
        segments = list(manifest.get("segments", []) or [])
        if not segments:
            raise RuntimeError("no recorded segments found in session manifest")

        if self.args.segment_id:
            segment = None
            for item in segments:
                if str(item.get("segment_id", "") or "") == self.args.segment_id:
                    segment = item
                    break
            if segment is None:
                raise RuntimeError(f"segment not found: {self.args.segment_id}")
            clip_start_wall_ts = float(segment.get("wall_ts_start", 0.0) or 0.0) + self.args.offset_seconds
        else:
            clip_start_wall_ts = self.args.clip_start_wall_ts

        target_segment = pick_segment_for_clip(
            segments,
            clip_start_wall_ts=clip_start_wall_ts,
            clip_duration_seconds=self.args.clip_duration_seconds,
        )
        segment_start = float(target_segment.get("wall_ts_start", 0.0) or 0.0)
        source_path = Path(str(target_segment.get("file_path", "") or "")).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"segment file missing: {source_path}")

        clips_dir = session_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_id = time.strftime("clip-%Y%m%d-%H%M%S", time.localtime())
        output_path = Path(self.args.output_file).expanduser().resolve() if self.args.output_file else clips_dir / (
            clip_id + f".{self.args.output_container}"
        )
        relative_start = max(0.0, clip_start_wall_ts - segment_start)
        await self._run_ffmpeg_clip(
            input_path=source_path,
            output_path=output_path,
            start_seconds=relative_start,
            duration_seconds=self.args.clip_duration_seconds,
        )
        print(f"[{_ts()}] clip exported: {output_path}")

        if self.args.build_srt:
            asr_jsonl_path = session_dir / "timeline" / "asr.jsonl"
            asr_events = load_jsonl(asr_jsonl_path)
            srt_text = build_srt_text(
                asr_events,
                clip_start_wall_ts=clip_start_wall_ts,
                clip_duration_seconds=self.args.clip_duration_seconds,
            )
            srt_path = output_path.with_suffix(".srt")
            srt_path.write_text(srt_text, encoding="utf-8")
            print(f"[{_ts()}] subtitle written: {srt_path}")

        return 0

    async def _resolve_real_room_id(self, http: BiliHttpClient, room_id: int) -> int:
        try:
            return await http.resolve_real_room_id(room_id=room_id, cookie=self.cookie)
        except Exception as e:
            print(f"[{_ts()}] WARN resolve real room id failed: {e!r}")
            return room_id

    async def _start_ws(self, room_id: int):
        assert self._http is not None
        self._ws_client = DanmakuRealtimeClient(
            http_client=self._http,
            room_id=room_id,
            cookie=self.cookie,
            wbi_cookie=self.wbi_cookie,
            ws_require_wbi_sign=self.args.wbi_sign_enabled,
            prefer_buvid3_ws_cookie=self.args.allow_buvid3_only,
            on_danmaku=self._on_ws_event,
        )
        await self._ws_client.start()
        print(f"[{_ts()}] INFO realtime ws started")

    async def _start_audio_asr(self, room_id: int):
        self._asr_worker = build_asr_worker_or_none(
            model_dir=self.args.asr_model_dir,
            sample_rate=self.args.audio_sample_rate,
            threads=self.args.asr_threads,
            vad_model_path=self.args.asr_vad_model_path,
            vad_threshold=self.args.asr_vad_threshold,
            vad_min_silence_duration=self.args.asr_vad_min_silence_duration,
            vad_min_speech_duration=self.args.asr_vad_min_speech_duration,
            vad_max_speech_duration=self.args.asr_vad_max_speech_duration,
            sense_voice_language=self.args.asr_sense_voice_language,
            sense_voice_use_itn=self.args.asr_sense_voice_use_itn,
        )
        if self._asr_worker is None:
            print(f"[{_ts()}] WARN asr unavailable, skip ASR timeline capture")
            return
        self._audio_task = asyncio.create_task(
            self._audio_loop(room_id),
            name="smartclip-audio-asr",
        )
        print(f"[{_ts()}] INFO asr enabled")

    async def _capture_segments_loop(self, room_id: int):
        assert self._http is not None
        assert self._layout is not None
        deadline = time.time() + max(1, int(self.args.duration_seconds))
        segment_index = 0
        backoff = 1.0

        while (time.time() < deadline) and (not self._stop_event.is_set()):
            segment_index += 1
            remaining = max(1.0, deadline - time.time())
            segment_duration = min(float(self.args.segment_seconds), remaining)
            segment_id = f"segment-{segment_index:04d}"
            segment_path = _segment_path(self._layout, segment_index, self.args.output_container)
            try:
                urls = await self._http.get_room_play_urls(
                    room_id=room_id,
                    cookie=self.cookie,
                    pull_protocol=self.args.audio_pull_protocol,
                    api_preference=self.args.audio_pull_api_preference,
                )
                if not urls:
                    raise RuntimeError("no play url available")
                stream_url = urls[0]
                record = await self._record_single_segment(
                    segment_id=segment_id,
                    stream_url=stream_url,
                    output_path=segment_path,
                    duration_seconds=segment_duration,
                    room_id=room_id,
                )
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                record = SegmentRecord(
                    segment_id=segment_id,
                    file_path=str(segment_path),
                    wall_ts_start=time.time(),
                    wall_ts_end=time.time(),
                    duration_seconds=0.0,
                    stream_url="",
                    ok=False,
                    returncode=-1,
                    error=str(e),
                )
                print(f"[{_ts()}] WARN recording failed: {e!r}")
                await asyncio.sleep(backoff)
                backoff = min(5.0, backoff * 2.0)

            self._append_segment_record(record)

        print(f"[{_ts()}] capture loop finished")

    async def _record_single_segment(
        self,
        *,
        segment_id: str,
        stream_url: str,
        output_path: Path,
        duration_seconds: float,
        room_id: int,
    ) -> SegmentRecord:
        header_blob = ""
        if self.args.audio_http_headers_enabled:
            header_blob = AudioRequestOptions.for_room(
                room_id=room_id,
                user_agent=DEFAULT_UA,
                cookie=self.cookie,
            ).build_header_blob()
        cmd = [
            self.args.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
        ]
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

        print(f"[{_ts()}] RECORD {segment_id} duration={duration_seconds:.1f}s -> {output_path.name}")
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
        ok = returncode == 0 and output_path.exists()
        if ok:
            print(
                f"[{_ts()}] RECORD {segment_id} done duration={wall_ts_end - wall_ts_start:.1f}s "
                f"file={output_path.name}",
                flush=True,
            )
        else:
            print(f"[{_ts()}] WARN RECORD {segment_id} failed rc={returncode} err={error or '-'}", flush=True)
        return SegmentRecord(
            segment_id=segment_id,
            file_path=str(output_path),
            wall_ts_start=wall_ts_start,
            wall_ts_end=wall_ts_end,
            duration_seconds=max(0.0, wall_ts_end - wall_ts_start),
            stream_url=stream_url,
            ok=ok,
            returncode=returncode,
            error=error,
        )

    def _append_segment_record(self, record: SegmentRecord):
        segments = self._manifest.setdefault("segments", [])
        assert isinstance(segments, list)
        segments.append(asdict(record))
        self._flush_manifest()
        if self._timeline_events_path is not None:
            append_jsonl(
                self._timeline_events_path,
                {
                    "event_type": "recording_segment",
                    "wall_ts_start": record.wall_ts_start,
                    "wall_ts_end": record.wall_ts_end,
                    "segment_id": record.segment_id,
                    "file_path": record.file_path,
                    "ok": record.ok,
                    "error": record.error,
                },
            )

    def _flush_manifest(self):
        assert self._manifest_path is not None
        write_json(self._manifest_path, self._manifest)

    async def _shutdown_capture(self):
        if self._history_task is not None:
            self._history_task.cancel()
        if self._audio_task is not None:
            self._audio_task.cancel()
        await asyncio.gather(
            *(task for task in (self._history_task, self._audio_task) if task is not None),
            return_exceptions=True,
        )
        self._history_task = None
        self._audio_task = None

        if self._asr_worker is not None:
            try:
                for seg in self._asr_worker.flush():
                    self._record_asr_segment(seg, source="flush")
            except Exception:
                pass
        self._asr_worker = None

        if self._ws_client is not None:
            await self._ws_client.stop()
            self._ws_client = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._http = None

        self._manifest["ended_at"] = time.time()
        self._flush_manifest()
        self._write_summary_report()

    def _write_summary_report(self):
        assert self._layout is not None
        segments = list(self._manifest.get("segments", []) or [])
        ok_segments = [item for item in segments if item.get("ok")]
        report_path = self._layout.reports_dir / "capture_summary.md"
        lines = [
            "# Smart Clip Capture Summary",
            "",
            f"- Session: `{self._layout.root}`",
            f"- Requested room: `{self.args.room_id}`",
            f"- Real room: `{self._manifest.get('real_room_id', self.args.room_id)}`",
            f"- Segment count: `{len(segments)}`",
            f"- Successful segments: `{len(ok_segments)}`",
            f"- Timeline file: `{self._layout.timeline_dir / 'events.jsonl'}`",
            f"- ASR file: `{self._layout.timeline_dir / 'asr.jsonl'}`",
            "",
            "## Suggested Manual Checks",
            "",
            "- Open one segment locally and确认画面、声音可播放。",
            "- 检查 `timeline/events.jsonl` 中是否同时出现 `recording_segment`、`danmaku`、`asr` 事件。",
            "- 选一个单段内的时间窗，运行 `clip` 子命令导出片段，再检查 `srt` 是否大致对齐。",
            "",
            "## Example Commands",
            "",
            f"- `python3 live_smart_clip_pipeline_runner.py clip --session-dir {self._layout.root} --segment-id segment-0001 --offset-seconds 10 --clip-duration-seconds 20 --build-srt`",
        ]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{_ts()}] summary report: {report_path}")

    async def _on_ws_event(self, item: DanmakuItem):
        payload = {
            "event_type": "danmaku",
            "source": "ws",
            "uid": item.uid,
            "nickname": item.nickname,
            "text": item.text,
            "timeline": item.timeline,
            "received_wall_ts": time.time(),
        }
        self._append_timeline_event(payload)
        print(f"[{_ts()}] DANMU {item.nickname or item.uid or '观众'}: {item.text}", flush=True)

    async def _poll_history_loop(self, room_id: int):
        assert self._http is not None
        interval = max(2, int(self.args.poll_interval))
        seen: set[str] = set()
        while not self._stop_event.is_set():
            try:
                items = await self._http.get_history_danmaku(room_id=room_id, cookie=self.cookie)
                for item in items:
                    key = item.dedup_key or f"{item.uid}|{item.timeline}|{item.text}"
                    if key in seen:
                        continue
                    seen.add(key)
                    payload = {
                        "event_type": "danmaku",
                        "source": "history",
                        "uid": item.uid,
                        "nickname": item.nickname,
                        "text": item.text,
                        "timeline": item.timeline,
                        "received_wall_ts": time.time(),
                    }
                    self._append_timeline_event(payload)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[{_ts()}] WARN poll history failed: {e!r}", flush=True)
                await asyncio.sleep(interval)

    async def _audio_loop(self, room_id: int):
        assert self._http is not None
        assert self._asr_worker is not None
        request_options = None
        if self.args.audio_http_headers_enabled:
            request_options = AudioRequestOptions.for_room(
                room_id=room_id,
                user_agent=DEFAULT_UA,
                cookie=self.cookie,
            )
        header_blob = request_options.build_header_blob() if request_options is not None else ""

        while not self._stop_event.is_set():
            proc = None
            try:
                urls = await self._http.get_room_play_urls(
                    room_id=room_id,
                    cookie=self.cookie,
                    pull_protocol=self.args.audio_pull_protocol,
                    api_preference=self.args.audio_pull_api_preference,
                )
                if not urls:
                    raise RuntimeError("no audio play url available")
                cmd = [
                    self.args.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                ]
                if header_blob:
                    cmd.extend(["-headers", header_blob])
                cmd.extend(
                    [
                        "-i",
                        urls[0],
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(self.args.audio_sample_rate),
                        "-f",
                        "s16le",
                        "pipe:1",
                    ]
                )
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                assert proc.stdout is not None
                bytes_per_second = max(1, int(self.args.audio_sample_rate) * 2)
                chunk_bytes = max(320, int(bytes_per_second * 0.1))
                while not self._stop_event.is_set():
                    chunk = await proc.stdout.read(chunk_bytes)
                    if not chunk:
                        break
                    segments = self._asr_worker.feed_pcm(chunk)
                    self._drain_asr_events()
                    for seg in segments:
                        self._record_asr_segment(seg, source="stream")
                raise RuntimeError("audio stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[{_ts()}] WARN audio/asr failed: {e!r}", flush=True)
                self._restart_asr_stream(str(e))
                await asyncio.sleep(1.0)
            finally:
                if proc is not None and proc.returncode is None:
                    proc.terminate()
                    await asyncio.gather(proc.wait(), return_exceptions=True)

    def _drain_asr_events(self):
        if self._asr_worker is None:
            return
        drain = getattr(self._asr_worker, "drain_events", None)
        if not callable(drain):
            return
        for event in drain() or []:
            if not isinstance(event, ASRDebugEvent):
                continue
            self._append_timeline_event(
                {
                    "event_type": "asr_debug",
                    "source": "asr",
                    "kind": event.kind,
                    "message": event.message,
                    "wall_ts": event.wall_ts,
                    "ts_start": event.ts_start,
                    "ts_end": event.ts_end,
                    "text": event.text,
                }
            )

    def _restart_asr_stream(self, reason: str):
        if self._asr_worker is None:
            return
        restart = getattr(self._asr_worker, "restart_stream", None)
        if not callable(restart):
            return
        try:
            segments = restart(flush_partial=True, reason=reason)
        except Exception:
            return
        for seg in segments or []:
            self._record_asr_segment(seg, source="restart")
        self._drain_asr_events()

    def _record_asr_segment(self, seg: ASRSegment, *, source: str):
        payload = {
            "event_type": "asr",
            "source": source,
            "text": seg.text,
            "ts_start": seg.ts_start,
            "ts_end": seg.ts_end,
            "wall_ts_start": seg.wall_ts_start,
            "wall_ts_end": seg.wall_ts_end,
            "conf": seg.conf,
        }
        self._append_timeline_event(payload)
        if self._asr_jsonl_path is not None:
            append_jsonl(self._asr_jsonl_path, payload)
        print(
            f"[{_ts()}] ASR ({seg.ts_start:.1f}-{seg.ts_end:.1f}s | wall={seg.wall_ts_start:.1f}-{seg.wall_ts_end:.1f}) {seg.text}",
            flush=True,
        )

    def _append_timeline_event(self, payload: dict[str, object]):
        if self._timeline_events_path is None:
            return
        append_jsonl(self._timeline_events_path, payload)

    async def _run_ffmpeg_clip(
        self,
        *,
        input_path: Path,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ):
        cmd = [
            self.args.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-c",
            "copy",
            str(output_path),
        ]
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
        if returncode != 0 or not output_path.exists():
            raise RuntimeError(f"ffmpeg clip export failed rc={returncode} err={' | '.join(stderr_lines[-3:])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="人类手工验证：B站直播录播 / 时间轴 / 本地切片单脚本，不包含投稿测试"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_cookie_args(p: argparse.ArgumentParser):
        p.add_argument("--cookie", default="", help="B站 cookie 字符串")
        p.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="cookie 文件路径")
        p.add_argument(
            "--cookie-from-plugin-config",
            action="store_true",
            help="从插件配置读取 cookie，优先 user_auth.bili_cookie，再退回 bili_login_cookie",
        )
        p.add_argument(
            "--plugin-config-file",
            default=DEFAULT_PLUGIN_CONFIG_FILE,
            help="插件配置文件路径，仅在 --cookie-from-plugin-config 时使用",
        )
        p.add_argument("--wbi-cookie", default="", help="WS鉴权专用 cookie，不填则复用 --cookie")

    probe = subparsers.add_parser("probe", help="验证房间可解析、可获取播放地址")
    probe.add_argument("--room-id", type=int, default=DEFAULT_ROOM_ID, help="直播间号，默认 27004785")
    probe.add_argument(
        "--audio-pull-protocol",
        choices=("http_flv", "hls"),
        default="http_flv",
        help="拉流协议",
    )
    probe.add_argument(
        "--audio-pull-api-preference",
        choices=("getRoomPlayInfo", "playUrl"),
        default="getRoomPlayInfo",
        help="播放地址 API 优先级",
    )
    add_cookie_args(probe)

    capture = subparsers.add_parser("capture", help="录制分段并采集弹幕 / ASR 时间轴")
    capture.add_argument("--room-id", type=int, default=DEFAULT_ROOM_ID, help="直播间号，默认 27004785")
    capture.add_argument("--duration-seconds", type=int, default=180, help="总采集时长")
    capture.add_argument("--segment-seconds", type=int, default=60, help="单段录制时长")
    capture.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    capture.add_argument(
        "--output-container",
        choices=("mkv", "mp4"),
        default="mkv",
        help="录制容器格式",
    )
    capture.add_argument("--ffmpeg-path", default="ffmpeg", help="ffmpeg 可执行文件")
    capture.add_argument(
        "--audio-pull-protocol",
        choices=("http_flv", "hls"),
        default="http_flv",
        help="播放地址协议",
    )
    capture.add_argument(
        "--audio-pull-api-preference",
        choices=("getRoomPlayInfo", "playUrl"),
        default="getRoomPlayInfo",
        help="播放地址 API 优先级",
    )
    capture.add_argument(
        "--audio-http-headers-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ffmpeg 请求是否注入 Referer / Origin / User-Agent / Cookie",
    )
    capture.add_argument("--poll-interval", type=int, default=8, help="history 轮询间隔")
    capture.add_argument("--no-ws", action="store_true", help="禁用实时弹幕 WS")
    capture.add_argument("--no-history", action="store_true", help="禁用 gethistory 回补")
    capture.add_argument("--asr", action="store_true", help="启用 ASR 时间轴采集")
    capture.add_argument(
        "--allow-buvid3-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="WS 鉴权优先仅携带 buvid3",
    )
    capture.add_argument(
        "--wbi-sign-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否为 getDanmuInfo 启用 WBI 签名",
    )
    capture.add_argument("--audio-sample-rate", type=int, default=16000, help="ASR 音频采样率")
    capture.add_argument("--asr-model-dir", default=DEFAULT_ASR_MODEL_DIR, help="sherpa 模型目录")
    capture.add_argument("--asr-vad-model-path", default=DEFAULT_ASR_VAD_MODEL_PATH, help="VAD 模型路径")
    capture.add_argument("--asr-vad-threshold", type=float, default=0.3, help="VAD 阈值")
    capture.add_argument("--asr-vad-min-silence-duration", type=float, default=0.35, help="VAD 句尾静音秒数")
    capture.add_argument("--asr-vad-min-speech-duration", type=float, default=0.25, help="VAD 最小语音时长")
    capture.add_argument("--asr-vad-max-speech-duration", type=float, default=20.0, help="VAD 最长语音时长")
    capture.add_argument("--asr-sense-voice-language", default="auto", help="SenseVoice 语言参数")
    capture.add_argument(
        "--asr-sense-voice-use-itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SenseVoice 是否启用 ITN",
    )
    capture.add_argument("--asr-threads", type=int, default=1, help="ASR 线程数")
    add_cookie_args(capture)

    clip = subparsers.add_parser("clip", help="基于单段录制导出本地 clip，可选生成 SRT")
    clip.add_argument("--session-dir", required=True, help="capture 生成的 session 目录")
    clip.add_argument("--segment-id", default="", help="目标 segment_id，例如 segment-0001")
    clip.add_argument("--offset-seconds", type=float, default=0.0, help="相对 segment 起点偏移秒数")
    clip.add_argument("--clip-start-wall-ts", type=float, default=0.0, help="绝对 wall clock 起点")
    clip.add_argument("--clip-duration-seconds", type=float, default=20.0, help="导出 clip 时长")
    clip.add_argument("--output-file", default="", help="输出文件路径")
    clip.add_argument(
        "--output-container",
        choices=("mkv", "mp4"),
        default="mkv",
        help="导出 clip 容器格式",
    )
    clip.add_argument("--ffmpeg-path", default="ffmpeg", help="ffmpeg 可执行文件")
    clip.add_argument("--build-srt", action="store_true", help="为 clip 生成同名 srt")

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = LiveSmartClipPipelineRunner(args)
    loop = asyncio.get_running_loop()

    stop_once = {"done": False}

    def _signal_stop():
        if stop_once["done"]:
            return
        stop_once["done"] = True
        asyncio.create_task(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_stop)
        except NotImplementedError:
            pass

    return await app.run()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

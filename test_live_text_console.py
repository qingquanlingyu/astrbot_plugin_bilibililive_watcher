#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import time
from pathlib import Path

import aiohttp

try:  # pragma: no cover
    from .asr_sherpa import ASRDebugEvent, build_asr_worker_or_none
    from .audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from .bili_auth import extract_buvid3
    from .bili_http import BiliHttpClient, DEFAULT_UA
    from .bili_ws import DanmakuRealtimeClient
    from .models import ASRSegment, DanmakuItem
except ImportError:  # pragma: no cover
    from asr_sherpa import ASRDebugEvent, build_asr_worker_or_none
    from audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from bili_auth import extract_buvid3
    from bili_http import BiliHttpClient, DEFAULT_UA
    from bili_ws import DanmakuRealtimeClient
    from models import ASRSegment, DanmakuItem

DEFAULT_COOKIE_FILE = "~/.bilibili-cookie.json"
DEFAULT_ASR_MODEL_DIR = (
    "./models/sherpa/rknn/"
    "sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
DEFAULT_ASR_VAD_MODEL_PATH = "./models/vad/silero_vad.onnx"


def _ts() -> str:
    return time.strftime("%H:%M:%S", time.localtime())


def _load_cookie(raw_cookie: str, cookie_file: str) -> str:
    if raw_cookie.strip():
        return raw_cookie.strip()
    path = Path(cookie_file).expanduser()
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("cookie", "") or "").strip()
    except Exception:
        return ""


class LiveTextConsole:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cookie = _load_cookie(args.cookie, args.cookie_file)
        self.wbi_cookie = self._resolve_wbi_cookie()
        self._stop_event = asyncio.Event()
        self._session: aiohttp.ClientSession | None = None
        self._http: BiliHttpClient | None = None
        self._ws_client: DanmakuRealtimeClient | None = None
        self._seen: dict[str, float] = {}
        self._asr_worker = None
        self._audio_task: asyncio.Task | None = None
        self._audio_conn_seq = 0
        self._current_audio_conn = 0
        self._pcm_total_chunks = 0
        self._pcm_total_bytes = 0
        self._pcm_conn_start_ts = 0.0
        self._pcm_last_chunk_ts = 0.0
        self._pcm_last_heartbeat_ts = 0.0

    async def run(self):
        self._session = aiohttp.ClientSession(headers={"User-Agent": DEFAULT_UA})
        self._http = BiliHttpClient(self._session)
        room_id = await self._resolve_real_room_id(self.args.room_id)
        print(f"[{_ts()}] room_id={self.args.room_id} real_room_id={room_id}")

        await self._start_ws(room_id)
        await self._start_audio_asr(room_id)

        poll_task = asyncio.create_task(self._poll_history_loop(room_id), name="poll-history")
        try:
            await self._stop_event.wait()
        finally:
            poll_task.cancel()
            await asyncio.gather(poll_task, return_exceptions=True)
            await self._shutdown()

    async def stop(self):
        self._stop_event.set()

    async def _resolve_real_room_id(self, room_id: int) -> int:
        assert self._http is not None
        try:
            return await self._http.resolve_real_room_id(room_id=room_id, cookie=self.cookie)
        except Exception as e:
            print(f"[{_ts()}] WARN resolve real room id failed: {e}")
            return room_id

    async def _start_ws(self, room_id: int):
        ws_enabled = (not self.args.no_ws) and self.args.danmu_ws_auth_mode != "history_only"
        if not ws_enabled:
            print(f"[{_ts()}] INFO ws disabled by args, will poll gethistory")
            return
        assert self._http is not None
        self._ws_client = DanmakuRealtimeClient(
            http_client=self._http,
            room_id=room_id,
            cookie=self.cookie,
            wbi_cookie=self.wbi_cookie,
            ws_require_wbi_sign=self.args.wbi_sign_enabled,
            on_danmaku=self._on_ws_event,
        )
        await self._ws_client.start()
        print(f"[{_ts()}] INFO realtime ws started")

    async def _start_audio_asr(self, room_id: int):
        if not self.args.asr:
            print(f"[{_ts()}] INFO asr disabled by default, pass --asr to enable")
            return
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
            print(
                f"[{_ts()}] WARN asr unavailable, continue danmaku-only. "
                "请确认当前python环境可 import sherpa_onnx"
            )
            return
        self._audio_task = asyncio.create_task(
            self._audio_loop(room_id),
            name="audio-asr-loop",
        )
        print(f"[{_ts()}] INFO asr enabled")

    async def _shutdown(self):
        if self._audio_task:
            self._audio_task.cancel()
            await asyncio.gather(self._audio_task, return_exceptions=True)
            self._audio_task = None
        if self._asr_worker is not None:
            try:
                for seg in self._asr_worker.flush():
                    self._print_asr(seg)
            except Exception:
                pass
        self._asr_worker = None
        if self._ws_client:
            await self._ws_client.stop()
            self._ws_client = None
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._http = None
        print(f"[{_ts()}] stopped")

    async def _on_ws_event(self, item: DanmakuItem):
        key = item.dedup_key or f"{item.uid}|{item.timeline}|{item.text}"
        if key in self._seen:
            return
        self._seen[key] = time.time()
        self._prune_seen()
        prefix = f"{item.event_type.upper():>5}"
        who = item.nickname or item.uid or "观众"
        print(f"[{_ts()}] {prefix} {who}: {item.text}", flush=True)

    async def _poll_history_loop(self, room_id: int):
        assert self._http is not None
        interval = max(2, int(self.args.poll_interval))
        while True:
            try:
                should_poll = self.args.no_ws
                if self._ws_client is not None:
                    should_poll = should_poll or (not self._ws_client.connected)
                    if self._ws_client.fatal_error:
                        should_poll = True
                if should_poll:
                    items = await self._http.get_history_danmaku(room_id=room_id, cookie=self.cookie)
                    for item in items:
                        key = item.dedup_key or f"{item.uid}|{item.timeline}|{item.text}"
                        if key in self._seen:
                            continue
                        self._seen[key] = time.time()
                        who = item.nickname or item.uid or "观众"
                        print(f"[{_ts()}] HIST  {who}: {item.text}", flush=True)
                    if self._ws_client and self._ws_client.fatal_error:
                        print(f"[{_ts()}] WARN ws fatal: {self._ws_client.fatal_error}")
                self._prune_seen()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[{_ts()}] WARN poll history failed: {e}")
            await asyncio.sleep(interval)

    async def _audio_loop(self, room_id: int):
        assert self._http is not None
        assert self._asr_worker is not None
        backoff = 1
        while True:
            try:
                urls = await self._http.get_room_play_urls(
                    room_id=room_id,
                    cookie=self.cookie,
                    pull_protocol=self.args.audio_pull_protocol,
                    api_preference=self.args.audio_pull_api_preference,
                )
                if not urls:
                    raise RuntimeError("no playurl")
                stream_url = urls[0]
                self._mark_audio_connect(stream_url)
                worker = AudioCaptureWorker(
                    ffmpeg_path=self.args.ffmpeg_path,
                    sample_rate=self.args.audio_sample_rate,
                )
                request_options = None
                if self.args.audio_http_headers_enabled:
                    request_options = AudioRequestOptions.for_room(
                        room_id=room_id,
                        user_agent=DEFAULT_UA,
                        cookie=self.cookie,
                    )
                await worker.run(
                    stream_url,
                    self._on_pcm,
                    request_options=request_options,
                )
                raise RuntimeError("audio stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._restart_asr_stream(f"audio capture retry after error: {e}")
                print(f"[{_ts()}] WARN audio/asr failed: {e}")
            await asyncio.sleep(backoff)
            backoff = min(8, backoff * 2)

    async def _on_pcm(self, pcm_chunk: bytes):
        assert self._asr_worker is not None
        self._mark_pcm_activity(len(pcm_chunk))
        try:
            segments = self._asr_worker.feed_pcm(pcm_chunk)
        except Exception as e:
            print(f"[{_ts()}] WARN asr feed failed: {e}")
            return
        self._print_asr_events()
        for seg in segments:
            self._print_asr(seg)

    def _restart_asr_stream(self, reason: str):
        if self._asr_worker is None:
            return
        restart = getattr(self._asr_worker, "restart_stream", None)
        if not callable(restart):
            return
        try:
            segments = restart(flush_partial=True, reason=reason)
        except Exception as e:
            print(f"[{_ts()}] WARN asr restart failed: {e}")
            return
        for seg in segments or []:
            self._print_asr(seg)
        self._print_asr_events()

    def _print_asr_events(self):
        if self._asr_worker is None:
            return
        drain = getattr(self._asr_worker, "drain_events", None)
        if not callable(drain):
            return
        for event in drain() or []:
            if not isinstance(event, ASRDebugEvent):
                continue
            print(f"[{_ts()}] VAD-EVENT {event.kind}: {event.message}", flush=True)

    def _mark_audio_connect(self, stream_url: str):
        self._audio_conn_seq += 1
        self._current_audio_conn = self._audio_conn_seq
        self._pcm_total_chunks = 0
        self._pcm_total_bytes = 0
        self._pcm_conn_start_ts = 0.0
        self._pcm_last_chunk_ts = 0.0
        self._pcm_last_heartbeat_ts = 0.0
        print(
            f"[{_ts()}] PCM-CONNECT conn={self._current_audio_conn} "
            f"protocol={self.args.audio_pull_protocol} url={stream_url}",
            flush=True,
        )

    def _mark_pcm_activity(self, chunk_bytes: int):
        if chunk_bytes <= 0:
            return
        now = time.time()
        if self._current_audio_conn <= 0:
            self._audio_conn_seq += 1
            self._current_audio_conn = self._audio_conn_seq
        if self._pcm_total_chunks <= 0:
            self._pcm_conn_start_ts = now
            self._pcm_last_heartbeat_ts = now
            print(
                f"[{_ts()}] PCM-FIRST conn={self._current_audio_conn} bytes={chunk_bytes}",
                flush=True,
            )
        else:
            gap = now - self._pcm_last_chunk_ts
            gap_threshold = max(0.0, float(getattr(self.args, "pcm_gap_seconds", 0.0) or 0.0))
            if gap_threshold > 0 and gap >= gap_threshold:
                print(
                    f"[{_ts()}] PCM-GAP conn={self._current_audio_conn} gap={gap:.1f}s bytes={chunk_bytes}",
                    flush=True,
                )
        self._pcm_total_chunks += 1
        self._pcm_total_bytes += chunk_bytes
        heartbeat_seconds = max(
            0.0,
            float(getattr(self.args, "pcm_heartbeat_seconds", 0.0) or 0.0),
        )
        if (
            heartbeat_seconds > 0
            and self._pcm_conn_start_ts > 0
            and (now - self._pcm_last_heartbeat_ts) >= heartbeat_seconds
        ):
            print(
                f"[{_ts()}] PCM-HEARTBEAT conn={self._current_audio_conn} "
                f"chunks={self._pcm_total_chunks} bytes={self._pcm_total_bytes} "
                f"uptime={now - self._pcm_conn_start_ts:.1f}s",
                flush=True,
            )
            self._pcm_last_heartbeat_ts = now
        self._pcm_last_chunk_ts = now

    def _print_asr(self, seg: ASRSegment):
        text = seg.text.strip()
        if not text:
            return
        print(
            (
                f"[{_ts()}] ASR   "
                f"({seg.ts_start:6.1f}-{seg.ts_end:6.1f}s"
                f" | wall={seg.wall_ts_start:10.1f}-{seg.wall_ts_end:10.1f}"
                f", conf={seg.conf:.2f}) {text}"
            ),
            flush=True,
        )

    def _prune_seen(self):
        now = time.time()
        cutoff = now - 600
        self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}

    def _resolve_wbi_cookie(self) -> str:
        explicit = self.args.wbi_cookie.strip()
        if explicit:
            return explicit
        if self.args.allow_buvid3_only:
            return extract_buvid3(self.cookie) or self.cookie
        return self.cookie


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="测试：持续打印 B站直播实时弹幕 + 可选 ASR 文本"
    )
    p.add_argument("room_id", type=int, help="直播间号（短号或真实房间号）")
    p.add_argument("--cookie", default="", help="B站 cookie 字符串")
    p.add_argument("--cookie-file", default=DEFAULT_COOKIE_FILE, help="cookie 文件路径")
    p.add_argument("--wbi-cookie", default="", help="WS鉴权专用cookie，不填则复用 --cookie")
    p.add_argument("--no-ws", action="store_true", help="禁用WS，仅使用 gethistory 轮询")
    p.add_argument(
        "--danmu-ws-auth-mode",
        choices=("signed_wbi", "unsigned", "history_only"),
        default="signed_wbi",
        help="与插件一致的弹幕WS模式；history_only 时强制仅轮询",
    )
    p.add_argument(
        "--allow-buvid3-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="与插件一致：WS鉴权优先仅携带 buvid3",
    )
    p.add_argument(
        "--wbi-sign-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="与插件一致：是否为 getDanmuInfo 启用 WBI 签名",
    )
    p.add_argument("--poll-interval", type=int, default=8, help="history 轮询间隔秒数")

    p.add_argument("--asr", action="store_true", help="启用 ASR（默认关闭）")
    p.add_argument(
        "--audio-pull-protocol",
        choices=("http_flv", "hls"),
        default="http_flv",
        help="音频拉流协议",
    )
    p.add_argument(
        "--audio-pull-api-preference",
        choices=("getRoomPlayInfo", "playUrl"),
        default="getRoomPlayInfo",
        help="与插件一致的音频拉流 API 优先级",
    )
    p.add_argument(
        "--audio-http-headers-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="与插件一致：ffmpeg 是否注入 Referer/Origin/User-Agent/Cookie",
    )
    p.add_argument("--ffmpeg-path", default="ffmpeg", help="ffmpeg 可执行文件")
    p.add_argument(
        "--pcm-gap-seconds",
        type=float,
        default=2.0,
        help="若连续两个 PCM chunk 的到达间隔超过该秒数，则打印 PCM-GAP 标记；设为 0 关闭",
    )
    p.add_argument(
        "--pcm-heartbeat-seconds",
        type=float,
        default=5.0,
        help="周期性打印 PCM-HEARTBEAT 标记，便于观察源流是否持续到达；设为 0 关闭",
    )
    p.add_argument("--audio-sample-rate", type=int, default=16000, help="音频采样率")
    p.add_argument("--asr-model-dir", default=DEFAULT_ASR_MODEL_DIR, help="sherpa 模型目录")
    p.add_argument(
        "--asr-vad-model-path",
        default=DEFAULT_ASR_VAD_MODEL_PATH,
        help="Silero VAD 模型文件",
    )
    p.add_argument("--asr-vad-threshold", type=float, default=0.3, help="Silero VAD 阈值")
    p.add_argument(
        "--asr-vad-min-silence-duration",
        type=float,
        default=0.35,
        help="VAD 判定一句结束所需最小静音秒数",
    )
    p.add_argument(
        "--asr-vad-min-speech-duration",
        type=float,
        default=0.25,
        help="VAD 判定为有效语音所需最小时长",
    )
    p.add_argument(
        "--asr-vad-max-speech-duration",
        type=float,
        default=20.0,
        help="VAD 单句最长时长（秒）",
    )
    p.add_argument(
        "--asr-sense-voice-language",
        default="auto",
        help="SenseVoice 语言参数：auto / zh / en / ja / ko / yue",
    )
    p.add_argument(
        "--asr-sense-voice-use-itn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SenseVoice 是否启用 ITN",
    )
    p.add_argument("--asr-threads", type=int, default=1, help="ASR线程数（RKNN 推荐 1）")
    return p.parse_args()


async def _main():
    args = parse_args()
    app = LiveTextConsole(args)
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

    await app.run()


if __name__ == "__main__":
    asyncio.run(_main())

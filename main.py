from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.entities import ProviderRequest

try:  # pragma: no cover
    from .asr_sherpa import SherpaASRWorker, build_asr_worker_or_none
    from .audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from .bili_auth import extract_buvid3
    from .bili_http import BiliHttpClient, DEFAULT_UA
    from .bili_ws import DanmakuRealtimeClient
    from .fusion import FusionEngine
    from .models import ASRSegment, DanmakuItem
    from .prompting import build_fused_prompt
except ImportError:  # pragma: no cover
    from asr_sherpa import SherpaASRWorker, build_asr_worker_or_none
    from audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from bili_auth import extract_buvid3
    from bili_http import BiliHttpClient, DEFAULT_UA
    from bili_ws import DanmakuRealtimeClient
    from fusion import FusionEngine
    from models import ASRSegment, DanmakuItem
    from prompting import build_fused_prompt

DEFAULT_COOKIE_FILE = "~/.bilibili-cookie.json"
DEFAULT_ASR_MODEL_DIR = (
    "./models/sherpa/rknn/"
    "sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
DEFAULT_ASR_VAD_MODEL_PATH = "./models/vad/silero_vad.onnx"
DEFAULT_CONVERSATION_CONTEXT_LIMIT = 12
DEFAULT_CONTEXT_WINDOW_MULTIPLIER = 3
DEFAULT_PCM_GAP_LOG_SECONDS = 2.0
DEFAULT_PCM_HEARTBEAT_SECONDS = 5.0
PLUGIN_DIR = Path(__file__).resolve().parent


@dataclass(slots=True)
class WatchConfig:
    enabled: bool
    debug: bool
    room_id: int
    reply_interval_seconds: int
    context_window_seconds: int
    danmaku_trigger_threshold: int
    asr_trigger_threshold: int
    target_umo: str
    target_platform_id: str
    target_type: str
    target_id: str
    max_reply_chars: int
    bilibili_cookie: str
    bilibili_cookie_file: str
    auto_load_cookie_from_file: bool
    pipeline_mode: str
    use_realtime_danmaku_ws: bool
    danmu_ws_auth_mode: str
    allow_buvid3_only: bool
    wbi_sign_enabled: bool
    audio_enabled: bool
    audio_pull_protocol: str
    audio_pull_api_preference: str
    audio_http_headers_enabled: bool
    ffmpeg_path: str
    audio_sample_rate: int
    asr_backend: str
    asr_strategy: str
    asr_model_dir: str
    asr_vad_model_path: str
    asr_vad_threshold: float
    asr_vad_min_silence_duration: float
    asr_vad_min_speech_duration: float
    asr_vad_max_speech_duration: float
    asr_sense_voice_language: str
    asr_sense_voice_use_itn: bool
    asr_runtime_probe_required: bool
    asr_threads: int
    asr_vad_enabled: bool
    asr_sentence_pause_seconds: float
    asr_sentence_min_chars: int
    singer_mode_enabled: bool
    singer_mode_threshold: float


@register(
    "astrbot_plugin_bilibililive_watcher",
    "YourName",
    "监听B站直播弹幕热度并触发模型生成短弹幕发送到指定会话。",
    "1.1.0",
)
class BilibiliLiveWatcherPlugin(Star):
    _LLM_REQUEST_SENTINEL_ATTR = "_bili_live_room_injection_applied"

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._http: BiliHttpClient | None = None
        self._buffer: list[DanmakuItem] = []
        self._asr_buffer: list[ASRSegment] = []
        self._context_danmaku_buffer: list[DanmakuItem] = []
        self._context_asr_buffer: list[ASRSegment] = []
        self._seen: dict[str, float] = {}
        self._last_trigger_ts = 0.0
        self._last_warn_ts = 0.0
        self._debug_enabled = False
        self._real_room_id: int | None = None
        self._real_room_id_source: int | None = None
        self._room_prompt_meta_room_id: int | None = None
        self._room_prompt_meta: dict[str, object] = {}
        self._runtime_room_id: int | None = None
        self._ws_client: DanmakuRealtimeClient | None = None
        self._ws_runtime_key: tuple | None = None
        self._ws_last_message_ts = 0.0
        self._history_bootstrapped = False
        self._audio_task: asyncio.Task | None = None
        self._audio_runtime_key: tuple | None = None
        self._asr_worker: SherpaASRWorker | None = None
        self._audio_conn_seq = 0
        self._current_audio_conn = 0
        self._pcm_total_chunks = 0
        self._pcm_total_bytes = 0
        self._pcm_conn_start_ts = 0.0
        self._pcm_last_chunk_ts = 0.0
        self._pcm_last_heartbeat_ts = 0.0
        self._fusion = FusionEngine()
        self._live_context_tool_turns: dict[str, float] = {}

    async def initialize(self):
        if self._task and not self._task.done():
            return
        self._session = aiohttp.ClientSession(headers={"User-Agent": DEFAULT_UA})
        self._http = BiliHttpClient(self._session)
        self._task = asyncio.create_task(self._watch_loop(), name="bili-live-watcher")
        logger.info("[bili_watcher] initialized")

    async def terminate(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        await self._stop_runtime_clients()

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        self._http = None
        logger.info("[bili_watcher] terminated")

    @filter.command("biliwatch help")
    async def biliwatch_help(self, event: AstrMessageEvent):
        lines = [
            "BiliWatch 可用指令：",
            "/biliwatch help - 查看本帮助",
            "/biliwatch status - 查看当前插件状态",
            "/biliwatch toggle [on|off] - 开关插件",
            "/biliwatch room <room_id> - 设置监听直播间",
            "/biliwatch bind - 绑定当前会话为发送目标",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("biliwatch status")
    async def biliwatch_status(self, event: AstrMessageEvent):
        cfg = self._load_config()
        target_umo = self._resolve_target_umo(cfg)
        platform_ids = self._get_available_platform_ids()
        last_trigger = "never"
        if self._last_trigger_ts > 0:
            last_trigger = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_trigger_ts))
        asr_status = "disabled"
        if self._asr_worker is not None:
            asr_status = self._asr_worker.status_text()
        ws_status = "disabled"
        if self._ws_client is not None:
            ws_status = f"connected={self._ws_client.connected} fatal={self._ws_client.fatal_error or '-'}"
        yield event.plain_result(
            "\n".join(
                [
                    f"enabled: {cfg.enabled}",
                    f"debug: {cfg.debug}",
                    f"pipeline_mode: {cfg.pipeline_mode}",
                    f"room_id: {cfg.room_id}",
                    f"target: {target_umo or '(未配置)'}",
                    f"platform_ids: {platform_ids or '[]'}",
                    f"reply_interval_seconds: {cfg.reply_interval_seconds}",
                    f"context_window_seconds: {cfg.context_window_seconds}",
                    f"danmaku_trigger_threshold: {cfg.danmaku_trigger_threshold}",
                    f"asr_trigger_threshold: {cfg.asr_trigger_threshold}",
                    f"pending_danmaku_buffer_size: {len(self._buffer)}",
                    f"pending_asr_buffer_size: {len(self._asr_buffer)}",
                    f"context_danmaku_buffer_size: {len(self._context_danmaku_buffer)}",
                    f"context_asr_buffer_size: {len(self._context_asr_buffer)}",
                    f"ws: {ws_status}",
                    f"asr: {asr_status}",
                    f"audio_task_running: {bool(self._audio_task and not self._audio_task.done())}",
                    f"last_trigger: {last_trigger}",
                ]
            )
        )

    @filter.command("biliwatch room", alias={"设置弹幕监听直播间"})
    async def biliwatch_set_room(self, event: AstrMessageEvent, room_id: str = ""):
        rid = str(room_id or "").strip()
        if not rid:
            parts = event.message_str.strip().split()
            if len(parts) >= 2:
                rid = parts[-1].strip()

        try:
            room_id_int = int(rid)
            if room_id_int <= 0:
                raise ValueError
        except Exception:
            yield event.plain_result("room_id 无效，请输入正整数。例如：/biliwatch room 22642754")
            return

        self._set_config_value("room_id", room_id_int)
        self._real_room_id = None
        self._real_room_id_source = None
        self._runtime_room_id = None
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        yield event.plain_result(f"已将监听直播间设置为 {room_id_int} {suffix}")

    @filter.command("biliwatch bind", alias={"绑定弹幕发送到当前会话"})
    async def biliwatch_bind_here(self, event: AstrMessageEvent):
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if not umo:
            yield event.plain_result("无法获取当前会话标识（unified_msg_origin 为空）")
            return

        self._set_config_value("target_umo", umo)
        parts = umo.split(":")
        if len(parts) >= 3:
            self._set_config_value("target_platform_id", parts[0].strip())
            msg_type = parts[1].strip().lower()
            if "friend" in msg_type or "private" in msg_type:
                self._set_config_value("target_type", "private")
            else:
                self._set_config_value("target_type", "group")
            self._set_config_value("target_id", ":".join(parts[2:]).strip())

        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        yield event.plain_result(f"已绑定发送目标到当前会话：{umo} {suffix}")

    @filter.command("biliwatch toggle", alias={"切换弹幕监听开关"})
    async def biliwatch_toggle(self, event: AstrMessageEvent, value: str = ""):
        raw = str(value or "").strip().lower()
        if not raw:
            parts = event.message_str.strip().split()
            if len(parts) >= 2:
                raw = parts[-1].strip().lower()

        current = self._to_bool(self.config.get("enabled", True), True)
        if raw in ("on", "enable", "enabled", "true", "1", "开", "开启"):
            new_value = True
        elif raw in ("off", "disable", "disabled", "false", "0", "关", "关闭"):
            new_value = False
        else:
            new_value = not current

        self._set_config_value("enabled", new_value)
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        state = "开启" if new_value else "关闭"
        yield event.plain_result(f"已{state} B站直播监听 {suffix}")

    @filter.llm_tool(name="bili_live_context_window")
    async def bili_live_context_window(self, event: AstrMessageEvent):
        """
        获取当前监听直播间在 context_window_seconds 窗口内的完整直播上下文，并按时间顺序返回。
        仅当插件已启用、直播间当前状态为“直播中”，且用户问题明显涉及当前直播正在发生的内容时调用。
        同一轮对话最多调用一次；如果问题无关、直播未开播、插件关闭或窗口为空，不要调用。
        """
        cfg = self._load_config()
        if not cfg.enabled:
            return self._dump_live_context_tool_result(
                available=False,
                reason="plugin_disabled",
                room_state=self._build_live_room_state_payload(cfg=cfg, room_id=cfg.room_id, room_meta={}),
            )
        if cfg.room_id <= 0:
            return self._dump_live_context_tool_result(
                available=False,
                reason="room_not_configured",
                room_state=self._build_live_room_state_payload(cfg=cfg, room_id=cfg.room_id, room_meta={}),
            )

        room_id = await self._resolve_real_room_id(cfg.room_id, cfg.bilibili_cookie)
        room_meta = await self._get_room_prompt_meta(room_id=room_id, cookie=cfg.bilibili_cookie)
        room_state = self._build_live_room_state_payload(cfg=cfg, room_id=room_id, room_meta=room_meta)
        if room_state["live_status"] != "直播中":
            return self._dump_live_context_tool_result(
                available=False,
                reason="room_not_live",
                room_state=room_state,
            )

        turn_key = self._build_live_context_tool_turn_key(event)
        self._prune_live_context_tool_turns()
        if turn_key and turn_key in self._live_context_tool_turns:
            return self._dump_live_context_tool_result(
                available=False,
                reason="already_provided_this_turn",
                room_state=room_state,
            )

        self._prune_old(cfg.context_window_seconds)
        ordered_context = self._build_ordered_context(
            danmaku_items=self._recent_context_danmaku_items(cfg),
            asr_segments=self._recent_context_asr_segments(cfg),
        )
        if not ordered_context:
            return self._dump_live_context_tool_result(
                available=False,
                reason="empty_context_window",
                room_state=room_state,
                window_seconds=cfg.context_window_seconds,
            )

        if turn_key:
            self._live_context_tool_turns[turn_key] = time.time()
        return self._dump_live_context_tool_result(
            available=True,
            reason="ok",
            room_state=room_state,
            window_seconds=cfg.context_window_seconds,
            ordered_context=ordered_context,
        )

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        cfg = self._load_config()
        if not cfg.enabled or cfg.room_id <= 0:
            return
        if self._llm_request_already_applied(req):
            return

        room_id = await self._resolve_real_room_id(cfg.room_id, cfg.bilibili_cookie)
        room_meta = await self._get_room_prompt_meta(room_id=room_id, cookie=cfg.bilibili_cookie)
        room_state = self._build_live_room_state_payload(cfg=cfg, room_id=room_id, room_meta=room_meta)
        payloads = [self._build_live_room_state_payload_text(room_state)]

        parts = self._ensure_extra_user_parts(req)
        if parts is None:
            logger.warning("[bili_watcher] request has no usable extra_user_content_parts, skip injection")
            return

        appended = False
        for payload in payloads:
            if not payload or self._has_existing_extra_user_payload(parts, payload):
                continue
            text_part = self._build_text_part(payload)
            if text_part is None:
                logger.warning("[bili_watcher] TextPart unavailable, skip request injection")
                return
            parts.append(text_part)
            appended = True

        if appended:
            logger.info(
                "[bili_watcher] injected live room request context room=%s live_status=%s umo=%s",
                room_state.get("room_id"),
                room_state.get("live_status"),
                str(getattr(event, "unified_msg_origin", "") or "unknown"),
            )
        self._mark_llm_request_applied(req)

    async def _watch_loop(self):
        while True:
            cfg = self._load_config()
            try:
                await self._tick(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[bili_watcher] tick failed: {e}")
            await asyncio.sleep(max(1, cfg.reply_interval_seconds))

    async def _tick(self, cfg: WatchConfig):
        self._debug_enabled = cfg.debug
        if not cfg.enabled:
            await self._stop_runtime_clients()
            return
        if cfg.room_id <= 0:
            self._log_warn_throttled("[bili_watcher] room_id 未配置，已跳过轮询")
            return

        target_umo = self._resolve_target_umo(cfg)
        if not target_umo:
            self._log_warn_throttled("[bili_watcher] 发送目标未配置，已跳过轮询")
            return

        room_id = await self._resolve_real_room_id(cfg.room_id, cfg.bilibili_cookie)
        await self._ensure_runtime(cfg, room_id)

        if self._should_poll_history(cfg):
            items = await self._fetch_history_danmaku(room_id, cfg.bilibili_cookie)
            self._ingest_history_batch(items)

        self._prune_old(cfg.context_window_seconds)
        trigger_blockers = self._get_trigger_blockers(cfg)
        if trigger_blockers:
            logger.warning("[bili_watcher] skip trigger: %s", " | ".join(trigger_blockers))
            return

        recent_danmaku = self._recent_context_danmaku_items(cfg)
        recent_asr = self._recent_context_asr_segments(cfg)
        reply = await self._generate_short_reply(
            cfg=cfg,
            target_umo=target_umo,
            room_id=room_id,
            danmaku_items=recent_danmaku,
            asr_segments=recent_asr,
        )
        self._last_trigger_ts = time.time()
        self._buffer.clear()
        self._asr_buffer.clear()

        if not reply:
            logger.warning("[bili_watcher] 模型未生成有效内容，本次不发送")
            return

        ok = await self.context.send_message(target_umo, MessageChain().message(reply))
        if ok:
            logger.info(f"[bili_watcher] sent to {target_umo}, room={room_id}, text={reply[:120]}")
        else:
            logger.warning(f"[bili_watcher] send_message failed, target={target_umo}")

    async def _ensure_runtime(self, cfg: WatchConfig, room_id: int):
        if self._runtime_room_id != room_id:
            await self._stop_runtime_clients()
            self._runtime_room_id = room_id
            self._ws_last_message_ts = 0.0
            self._history_bootstrapped = False
            self._room_prompt_meta_room_id = None
            self._room_prompt_meta = {}

        await self._ensure_ws_runtime(cfg, room_id)
        await self._ensure_audio_runtime(cfg, room_id)

    async def _ensure_ws_runtime(self, cfg: WatchConfig, room_id: int):
        if self._http is None:
            return

        ws_enabled = (
            cfg.pipeline_mode != "asr_only"
            and cfg.use_realtime_danmaku_ws
            and cfg.danmu_ws_auth_mode != "history_only"
        )
        if not ws_enabled:
            await self._stop_ws_runtime()
            return

        wbi_cookie = cfg.bilibili_cookie
        if cfg.allow_buvid3_only:
            wbi_cookie = extract_buvid3(cfg.bilibili_cookie) or cfg.bilibili_cookie
        ws_key = (
            room_id,
            cfg.bilibili_cookie,
            wbi_cookie,
            cfg.danmu_ws_auth_mode,
            cfg.wbi_sign_enabled,
        )
        if (
            self._ws_client is not None
            and self._ws_runtime_key == ws_key
            and self._ws_client.running
        ):
            return

        await self._stop_ws_runtime()
        self._ws_client = DanmakuRealtimeClient(
            http_client=self._http,
            room_id=room_id,
            cookie=cfg.bilibili_cookie,
            wbi_cookie=wbi_cookie,
            ws_require_wbi_sign=cfg.wbi_sign_enabled,
            on_danmaku=self._on_realtime_danmaku,
        )
        self._ws_runtime_key = ws_key
        await self._ws_client.start()

    async def _ensure_audio_runtime(self, cfg: WatchConfig, room_id: int):
        audio_enabled = cfg.pipeline_mode in ("danmu_plus_asr", "asr_only") and cfg.audio_enabled
        if not audio_enabled:
            await self._stop_audio_runtime()
            return

        audio_key = (
            room_id,
            cfg.pipeline_mode,
            cfg.bilibili_cookie,
            cfg.audio_pull_protocol,
            cfg.audio_pull_api_preference,
            cfg.audio_http_headers_enabled,
            cfg.ffmpeg_path,
            cfg.audio_sample_rate,
            cfg.asr_backend,
            cfg.asr_strategy,
            cfg.asr_model_dir,
            cfg.asr_vad_model_path,
            cfg.asr_vad_threshold,
            cfg.asr_vad_min_silence_duration,
            cfg.asr_vad_min_speech_duration,
            cfg.asr_vad_max_speech_duration,
            cfg.asr_sense_voice_language,
            cfg.asr_sense_voice_use_itn,
            cfg.asr_runtime_probe_required,
            cfg.asr_threads,
            cfg.asr_vad_enabled,
            cfg.asr_sentence_pause_seconds,
            cfg.asr_sentence_min_chars,
        )
        if self._audio_runtime_key == audio_key and self._audio_task and not self._audio_task.done():
            return

        await self._stop_audio_runtime()
        worker = self._build_asr_worker(cfg)
        if worker is None:
            self._audio_runtime_key = None
            return
        self._audio_runtime_key = audio_key
        self._asr_worker = worker

        self._audio_task = asyncio.create_task(
            self._audio_loop(cfg, room_id),
            name=f"bili-audio-asr-{room_id}",
        )

    async def _stop_runtime_clients(self):
        await self._stop_ws_runtime()
        await self._stop_audio_runtime()

    async def _stop_ws_runtime(self):
        if self._ws_client is not None:
            await self._ws_client.stop()
        self._ws_client = None
        self._ws_runtime_key = None
        self._ws_last_message_ts = 0.0

    async def _stop_audio_runtime(self):
        if self._audio_task is not None:
            self._audio_task.cancel()
            await asyncio.gather(self._audio_task, return_exceptions=True)
        self._audio_task = None
        self._audio_runtime_key = None
        self._reset_audio_observation()
        if self._asr_worker is not None:
            try:
                for seg in self._asr_worker.flush():
                    self._record_asr_segment(seg)
            except Exception:
                pass
        self._asr_worker = None

    def _build_asr_worker(self, cfg: WatchConfig) -> SherpaASRWorker | None:
        if cfg.asr_runtime_probe_required:
            return build_asr_worker_or_none(
                asr_strategy=cfg.asr_strategy,
                model_dir=cfg.asr_model_dir,
                sample_rate=cfg.audio_sample_rate,
                threads=cfg.asr_threads,
                vad_model_path=cfg.asr_vad_model_path,
                vad_threshold=cfg.asr_vad_threshold,
                vad_min_silence_duration=cfg.asr_vad_min_silence_duration,
                vad_min_speech_duration=cfg.asr_vad_min_speech_duration,
                vad_max_speech_duration=cfg.asr_vad_max_speech_duration,
                sense_voice_language=cfg.asr_sense_voice_language,
                sense_voice_use_itn=cfg.asr_sense_voice_use_itn,
                vad_enabled=cfg.asr_vad_enabled,
                sentence_pause_seconds=cfg.asr_sentence_pause_seconds,
                sentence_min_chars=cfg.asr_sentence_min_chars,
            )
        worker = SherpaASRWorker(
            asr_strategy=cfg.asr_strategy,
            model_dir=cfg.asr_model_dir,
            sample_rate=cfg.audio_sample_rate,
            threads=cfg.asr_threads,
            vad_model_path=cfg.asr_vad_model_path,
            vad_threshold=cfg.asr_vad_threshold,
            vad_min_silence_duration=cfg.asr_vad_min_silence_duration,
            vad_min_speech_duration=cfg.asr_vad_min_speech_duration,
            vad_max_speech_duration=cfg.asr_vad_max_speech_duration,
            sense_voice_language=cfg.asr_sense_voice_language,
            sense_voice_use_itn=cfg.asr_sense_voice_use_itn,
            vad_enabled=cfg.asr_vad_enabled,
            sentence_pause_seconds=cfg.asr_sentence_pause_seconds,
            sentence_min_chars=cfg.asr_sentence_min_chars,
        )
        return worker if worker.enabled else None

    async def _audio_loop(self, cfg: WatchConfig, room_id: int):
        assert self._http is not None
        assert self._asr_worker is not None

        backoff = 1
        while True:
            try:
                urls = await self._http.get_room_play_urls(
                    room_id=room_id,
                    cookie=cfg.bilibili_cookie,
                    pull_protocol=cfg.audio_pull_protocol,
                    api_preference=cfg.audio_pull_api_preference,
                )
                if not urls:
                    raise RuntimeError("no play url found")
                capture = AudioCaptureWorker(
                    ffmpeg_path=cfg.ffmpeg_path,
                    sample_rate=cfg.audio_sample_rate,
                )
                request_options = None
                if cfg.audio_http_headers_enabled:
                    request_options = AudioRequestOptions.for_room(
                        room_id=room_id,
                        user_agent=DEFAULT_UA,
                        cookie=cfg.bilibili_cookie,
                    )
                self._mark_audio_connect(cfg.audio_pull_protocol, urls[0])
                await capture.run(urls[0], self._on_pcm, request_options=request_options)
                raise RuntimeError("audio stream ended")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._restart_asr_stream(f"audio capture retry after error: {e}")
                logger.warning(f"[bili_watcher] audio/asr failed: {e}")
            await asyncio.sleep(backoff)
            backoff = min(8, backoff * 2)

    async def _on_realtime_danmaku(self, item: DanmakuItem):
        self._ws_last_message_ts = time.time()
        self._ingest_danmaku(item)

    async def _on_pcm(self, pcm_chunk: bytes):
        if self._asr_worker is None:
            return
        self._mark_pcm_activity(len(pcm_chunk))
        try:
            segments = self._asr_worker.feed_pcm(pcm_chunk)
        except Exception as e:
            logger.warning(f"[bili_watcher] asr feed failed: {e}")
            return
        for seg in segments:
            self._record_asr_segment(seg)
            self._debug_log_asr_segment(seg)

    def _restart_asr_stream(self, reason: str):
        if self._asr_worker is None:
            return
        restart = getattr(self._asr_worker, "restart_stream", None)
        if not callable(restart):
            return
        try:
            segments = restart(flush_partial=True, reason=reason)
        except Exception as e:
            logger.warning(f"[bili_watcher] ASR stream restart failed: {e}")
            return
        for seg in segments or []:
            self._record_asr_segment(seg)
            self._debug_log_asr_segment(seg)

    def _ingest_danmaku(self, item: DanmakuItem):
        key = item.dedup_key or f"{item.uid}|{item.timeline}|{item.text}"
        if key in self._seen:
            return
        self._seen[key] = time.time()
        self._buffer.append(item)
        self._context_danmaku_buffer.append(item)
        self._debug_log_danmaku(item)

    def _ingest_history_batch(self, items: list[DanmakuItem]) -> int:
        if not self._history_bootstrapped:
            seen_ts = time.time()
            for item in items:
                key = item.dedup_key or f"{item.uid}|{item.timeline}|{item.text}"
                self._seen[key] = seen_ts
            self._history_bootstrapped = True
            if items:
                logger.info(
                    "[bili_watcher] history bootstrap skipped %d backlog item(s)",
                    len(items),
                )
            return 0

        ingested = 0
        for item in items:
            before = len(self._buffer)
            self._ingest_danmaku(item)
            if len(self._buffer) > before:
                ingested += 1
        return ingested

    def _record_asr_segment(self, seg: ASRSegment):
        self._asr_buffer.append(seg)
        self._context_asr_buffer.append(seg)
        logger.info(
            "[bili_watcher] ASR segment emitted: text=%r audio=(%.2f-%.2f) wall=(%.2f-%.2f) conf=%.2f",
            seg.text,
            seg.ts_start,
            seg.ts_end,
            seg.wall_ts_start,
            seg.wall_ts_end,
            seg.conf,
        )

    def _reset_audio_observation(self):
        self._current_audio_conn = 0
        self._pcm_total_chunks = 0
        self._pcm_total_bytes = 0
        self._pcm_conn_start_ts = 0.0
        self._pcm_last_chunk_ts = 0.0
        self._pcm_last_heartbeat_ts = 0.0

    def _mark_audio_connect(self, protocol: str, stream_url: str):
        self._audio_conn_seq += 1
        self._current_audio_conn = self._audio_conn_seq
        self._pcm_total_chunks = 0
        self._pcm_total_bytes = 0
        self._pcm_conn_start_ts = 0.0
        self._pcm_last_chunk_ts = 0.0
        self._pcm_last_heartbeat_ts = 0.0
        logger.info(
            "[bili_watcher] PCM-CONNECT conn=%s protocol=%s url=%s",
            self._current_audio_conn,
            protocol,
            stream_url,
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
            logger.info(
                "[bili_watcher] PCM-FIRST conn=%s bytes=%s",
                self._current_audio_conn,
                chunk_bytes,
            )
        else:
            gap = now - self._pcm_last_chunk_ts
            if gap >= DEFAULT_PCM_GAP_LOG_SECONDS:
                logger.warning(
                    "[bili_watcher] PCM-GAP conn=%s gap=%.1fs bytes=%s",
                    self._current_audio_conn,
                    gap,
                    chunk_bytes,
                )
        self._pcm_total_chunks += 1
        self._pcm_total_bytes += chunk_bytes
        if (
            self._pcm_conn_start_ts > 0
            and (now - self._pcm_last_heartbeat_ts) >= DEFAULT_PCM_HEARTBEAT_SECONDS
        ):
            logger.info(
                "[bili_watcher] PCM-HEARTBEAT conn=%s chunks=%s bytes=%s uptime=%.1fs",
                self._current_audio_conn,
                self._pcm_total_chunks,
                self._pcm_total_bytes,
                now - self._pcm_conn_start_ts,
            )
            self._pcm_last_heartbeat_ts = now
        self._pcm_last_chunk_ts = now

    def _recent_context_danmaku_items(self, cfg: WatchConfig) -> list[DanmakuItem]:
        if not self._context_danmaku_buffer:
            return []
        return list(self._context_danmaku_buffer)

    def _recent_context_asr_segments(self, cfg: WatchConfig) -> list[ASRSegment]:
        if not self._context_asr_buffer:
            return []
        return list(self._context_asr_buffer)

    def _should_trigger_reply(self, cfg: WatchConfig) -> bool:
        return not self._get_trigger_blockers(cfg)

    async def _resolve_real_room_id(self, room_id: int, cookie: str) -> int:
        if self._real_room_id_source == room_id and self._real_room_id:
            return self._real_room_id
        if self._http is None:
            return room_id
        try:
            actual_id = await self._http.resolve_real_room_id(room_id=room_id, cookie=cookie)
            self._real_room_id = actual_id
            self._real_room_id_source = room_id
            return actual_id
        except Exception as e:
            logger.warning(f"[bili_watcher] resolve real room id failed: {e}")
        self._real_room_id = room_id
        self._real_room_id_source = room_id
        return room_id

    async def _fetch_history_danmaku(self, room_id: int, cookie: str) -> list[DanmakuItem]:
        if self._http is None:
            return []
        return await self._http.get_history_danmaku(room_id=room_id, cookie=cookie)

    def _should_poll_history(self, cfg: WatchConfig) -> bool:
        if cfg.pipeline_mode == "asr_only":
            return False
        if not cfg.use_realtime_danmaku_ws or cfg.danmu_ws_auth_mode == "history_only":
            return True
        if self._ws_client is None:
            return True
        if self._ws_client.fatal_error:
            return True
        if not self._ws_client.connected:
            return True
        if self._ws_last_message_ts <= 0:
            return True
        idle_threshold = max(15, cfg.reply_interval_seconds * 2)
        return (time.time() - self._ws_last_message_ts) >= idle_threshold

    async def _generate_short_reply(
        self,
        cfg: WatchConfig,
        target_umo: str,
        room_id: int,
        danmaku_items: list[DanmakuItem],
        asr_segments: list[ASRSegment],
    ) -> str:
        provider = self._resolve_provider(target_umo)
        if provider is None:
            logger.warning("[bili_watcher] no provider available")
            return ""

        conversation, contexts = await self._get_recent_contexts(
            target_umo=target_umo,
            max_messages=DEFAULT_CONVERSATION_CONTEXT_LIMIT,
        )
        system_prompt = await self._get_system_prompt(target_umo, conversation)
        room_meta = await self._get_room_prompt_meta(room_id=room_id, cookie=cfg.bilibili_cookie)
        contexts = self._inject_live_room_contexts(
            contexts=contexts,
            cfg=cfg,
            room_id=room_id,
            room_meta=room_meta,
        )
        fusion = self._fusion.build_summary(
            danmaku_items=danmaku_items,
            asr_segments=asr_segments,
            window_seconds=cfg.context_window_seconds,
            singer_mode_enabled=cfg.singer_mode_enabled,
            singer_mode_threshold=cfg.singer_mode_threshold,
        )
        fusion.ordered_context = self._build_ordered_context(
            danmaku_items=danmaku_items,
            asr_segments=asr_segments,
        )
        prompt = build_fused_prompt(
            room_id=room_id,
            room_title=room_meta.get("room_title", ""),
            anchor_name=room_meta.get("anchor_name", ""),
            fusion=fusion,
            max_reply_chars=cfg.max_reply_chars,
        )
        self._debug_log_prompt(prompt)

        text = await self._run_generation_once(
            provider=provider,
            prompt=prompt,
            contexts=contexts,
            system_prompt=system_prompt or "",
        )
        text = self._normalize_reply(text, cfg.max_reply_chars)
        if not text:
            return ""
        return text

    def _debug_log(self, message: str, *args):
        if not self._debug_enabled:
            return
        logger.info(message, *args)

    def _debug_log_danmaku(self, item: DanmakuItem):
        self._debug_log(
            "[bili_watcher][debug] danmaku: source=%s nickname=%s text=%r timeline=%s",
            item.source,
            item.nickname or item.uid or "观众",
            item.text,
            item.timeline,
        )

    def _debug_log_asr_segment(self, seg: ASRSegment):
        self._debug_log(
            "[bili_watcher][debug] asr: text=%r audio=(%.2f-%.2f) wall=(%.2f-%.2f) conf=%.2f",
            seg.text,
            seg.ts_start,
            seg.ts_end,
            seg.wall_ts_start,
            seg.wall_ts_end,
            seg.conf,
        )

    def _debug_log_prompt(self, prompt: str):
        self._debug_log("[bili_watcher][debug] prompt:\n%s", prompt)

    def _normalize_live_status(self, raw_status: object) -> str:
        if isinstance(raw_status, str):
            normalized = raw_status.strip()
            if normalized in {"直播中", "未开播", "状态未知"}:
                return normalized
            lowered = normalized.lower()
            if lowered in {"1", "live", "on", "streaming"}:
                return "直播中"
            if lowered in {"0", "offline", "off"}:
                return "未开播"
            return "状态未知"
        if raw_status is None:
            return "状态未知"
        try:
            value = int(raw_status)
        except (TypeError, ValueError):
            return "状态未知"
        if value > 0:
            return "直播中"
        if value == 0:
            return "未开播"
        return "状态未知"

    def _build_live_room_state_payload(
        self,
        *,
        cfg: WatchConfig,
        room_id: int,
        room_meta: dict[str, object] | None,
    ) -> dict[str, object]:
        meta = room_meta or {}
        return {
            "plugin_enabled": bool(cfg.enabled),
            "room_id": int(room_id or 0),
            "anchor_name": str(meta.get("anchor_name", "") or "").strip(),
            "room_title": str(meta.get("room_title", "") or "").strip(),
            "live_status": self._normalize_live_status(meta.get("live_status")),
        }

    def _build_live_room_state_context(self, room_state: dict[str, object]) -> dict[str, str]:
        return {
            "role": "system",
            "content": self._build_live_room_state_payload_text(room_state),
        }

    def _build_live_room_state_payload_text(self, room_state: dict[str, object]) -> str:
        return ("<bili_live_room_state>\n当有人问你是否在看直播，使用下述信息回答\n" + \
        f"正在观看直播\n主播名:{room_state['anchor_name']}\n状态:{room_state['live_status']}\n直播间名:{room_state['room_title']}\n" + \
        "</bili_live_room_state>\n")

    def _inject_live_room_contexts(
        self,
        *,
        contexts: list[dict],
        cfg: WatchConfig,
        room_id: int,
        room_meta: dict[str, object] | None,
    ) -> list[dict]:
        live_contexts = list(contexts or [])
        room_state = self._build_live_room_state_payload(cfg=cfg, room_id=room_id, room_meta=room_meta)
        live_contexts.append(self._build_live_room_state_context(room_state))
        return live_contexts

    async def _get_room_prompt_meta(self, room_id: int, cookie: str) -> dict[str, object]:
        if self._room_prompt_meta_room_id == room_id and self._room_prompt_meta:
            return dict(self._room_prompt_meta)
        if self._http is None:
            return {}
        try:
            meta = await self._http.get_room_prompt_meta(room_id=room_id, cookie=cookie)
            self._room_prompt_meta_room_id = room_id
            self._room_prompt_meta = {
                "room_title": meta.room_title,
                "anchor_name": meta.anchor_name,
                "live_status": self._normalize_live_status(getattr(meta, "live_status", None)),
            }
            return dict(self._room_prompt_meta)
        except Exception as e:
            logger.warning(f"[bili_watcher] get room prompt meta failed: {e}")
            return {}

    def _resolve_provider(self, target_umo: str):
        try:
            provider = self.context.get_using_provider(umo=target_umo)
        except TypeError:
            provider = self.context.get_using_provider()
        if not provider:
            provider = self.context.get_using_provider()
        return provider

    async def _run_generation_once(
        self,
        *,
        provider: object,
        prompt: str,
        contexts: list[dict],
        system_prompt: str,
    ) -> str:
        llm_resp = None
        try:
            llm_resp = await provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt,
            )
        except TypeError:
            try:
                llm_resp = await provider.text_chat(prompt=prompt, system_prompt=system_prompt)
            except TypeError:
                llm_resp = await provider.text_chat(prompt)
        except Exception as e:
            logger.warning(f"[bili_watcher] provider.text_chat failed: {e}")
            if self._is_tool_context_mismatch_error(e):
                llm_resp = await self._call_provider_without_contexts(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
        return self._extract_llm_text(llm_resp)

    async def _get_recent_contexts(self, target_umo: str, max_messages: int) -> tuple[object | None, list[dict]]:
        conv_mgr = getattr(self.context, "conversation_manager", None)
        if not conv_mgr:
            return None, []
        try:
            cid = await conv_mgr.get_curr_conversation_id(target_umo)
            if not cid:
                return None, []
            conv = await conv_mgr.get_conversation(target_umo, cid)
            if not conv:
                return None, []
            contexts = self._extract_contexts_from_conversation(conv)
            if max_messages > 0:
                contexts = contexts[-max_messages:]
            return conv, contexts
        except Exception as e:
            logger.warning(f"[bili_watcher] load conversation contexts failed: {e}")
            return None, []

    def _extract_contexts_from_conversation(self, conversation: object) -> list[dict]:
        history = getattr(conversation, "history", None)
        if history is None:
            return []
        raw_items: list = []
        if isinstance(history, str):
            try:
                parsed = json.loads(history)
                if isinstance(parsed, list):
                    raw_items = parsed
            except Exception:
                return []
        elif isinstance(history, list):
            raw_items = history
        else:
            return []

        contexts: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            if role not in ("user", "assistant", "system"):
                continue
            content = item.get("content", "")
            if isinstance(content, list):
                merged = []
                for c in content:
                    if isinstance(c, dict):
                        text = c.get("text")
                        if isinstance(text, str) and text.strip():
                            merged.append(text.strip())
                content_text = " ".join(merged).strip()
            else:
                content_text = str(content).strip()
            if role and content_text:
                contexts.append({"role": role, "content": content_text})
        return contexts

    async def _call_provider_without_contexts(self, provider: object, prompt: str, system_prompt: str) -> object | None:
        try:
            return await provider.text_chat(prompt=prompt, system_prompt=system_prompt)
        except TypeError:
            return await provider.text_chat(prompt)
        except Exception as e:
            logger.warning(f"[bili_watcher] retry without contexts failed: {e}")
            return None

    def _is_tool_context_mismatch_error(self, err: Exception) -> bool:
        text = str(err).lower()
        keywords = ("tool id() not found", "tool result's tool id", "tool_call_id", "tool result")
        return any(k in text for k in keywords)

    async def _get_system_prompt(self, umo: str, conversation: object | None) -> str:
        persona_mgr = getattr(self.context, "persona_manager", None)
        if not persona_mgr:
            return ""
        try:
            if conversation and getattr(conversation, "persona_id", None):
                persona = await persona_mgr.get_persona(conversation.persona_id)
                if persona and getattr(persona, "system_prompt", None):
                    return str(persona.system_prompt)
            try:
                default_persona = await persona_mgr.get_default_persona_v3(umo=umo)
            except TypeError:
                default_persona = await persona_mgr.get_default_persona_v3()
            if isinstance(default_persona, dict):
                return str(default_persona.get("prompt", "")).strip()
        except Exception as e:
            logger.warning(f"[bili_watcher] get system prompt failed: {e}")
        return ""

    def _build_ordered_context(
        self,
        *,
        danmaku_items: list[DanmakuItem],
        asr_segments: list[ASRSegment],
    ) -> list[dict]:
        merged: list[tuple[float, int, dict]] = []
        for item in danmaku_items:
            merged.append(
                (
                    item.ts,
                    0,
                    {
                        "source": "弹幕",
                        "speaker": item.nickname or item.uid or "观众",
                        "text": item.text,
                    },
                )
            )
        for seg in asr_segments:
            event_time = seg.wall_ts_end or seg.ts_end
            merged.append(
                (
                    event_time,
                    1,
                    {
                        "source": "主播",
                        "speaker": "主播",
                        "text": seg.text,
                    },
                )
            )
        merged.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in merged]

    def _build_live_context_tool_turn_key(self, event: AstrMessageEvent) -> str:
        if event is None:
            return ""
        parts: list[str] = []
        umo = str(getattr(event, "unified_msg_origin", "") or "").strip()
        if umo:
            parts.append(umo)
        for attr in ("message_id", "msg_id", "id"):
            value = getattr(event, attr, None)
            if value:
                parts.append(str(value).strip())
                break
        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None and len(parts) < 2:
            for attr in ("message_id", "msg_id", "id"):
                value = getattr(message_obj, attr, None)
                if value:
                    parts.append(str(value).strip())
                    break
        if len(parts) < 2:
            message_text = str(getattr(event, "message_str", "") or "").strip()
            if message_text:
                parts.append(message_text)
        return "|".join(parts)

    def _prune_live_context_tool_turns(self):
        if not self._live_context_tool_turns:
            return
        cutoff = time.time() - 3600
        self._live_context_tool_turns = {
            key: ts for key, ts in self._live_context_tool_turns.items() if ts >= cutoff
        }

    def _dump_live_context_tool_result(
        self,
        *,
        available: bool,
        reason: str,
        room_state: dict[str, object],
        window_seconds: int = 0,
        ordered_context: list[dict] | None = None,
    ) -> str:
        payload = {
            "available": bool(available),
            "reason": str(reason or ""),
            "room_state": room_state,
            "window_seconds": max(0, int(window_seconds or 0)),
            "ordered_context": list(ordered_context or []),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _llm_request_already_applied(cls, req: ProviderRequest) -> bool:
        try:
            return bool(getattr(req, cls._LLM_REQUEST_SENTINEL_ATTR, False))
        except Exception:
            return False

    @classmethod
    def _mark_llm_request_applied(cls, req: ProviderRequest) -> None:
        try:
            setattr(req, cls._LLM_REQUEST_SENTINEL_ATTR, True)
        except Exception:
            return

    @staticmethod
    def _ensure_extra_user_parts(req: ProviderRequest):
        parts = getattr(req, "extra_user_content_parts", None)
        if parts is None:
            parts = []
            try:
                setattr(req, "extra_user_content_parts", parts)
            except Exception:
                return None
        if isinstance(parts, list):
            return parts
        try:
            normalized = list(parts)
            setattr(req, "extra_user_content_parts", normalized)
            return normalized
        except Exception:
            return None

    @staticmethod
    def _has_existing_extra_user_payload(parts: list[object], payload: str) -> bool:
        for part in parts:
            if str(getattr(part, "text", "") or "") == payload:
                return True
        return False

    @staticmethod
    def _build_text_part(payload: str):
        try:
            from astrbot.core.agent.message import TextPart
        except Exception:
            return None
        return TextPart(text=payload)

    def _get_trigger_blockers(self, cfg: WatchConfig) -> list[str]:
        blockers: list[str] = []
        if cfg.pipeline_mode != "asr_only":
            danmaku_count = len(self._buffer)
            if danmaku_count < cfg.danmaku_trigger_threshold:
                blockers.append(
                    f"danmaku_count={danmaku_count} < danmaku_trigger_threshold={cfg.danmaku_trigger_threshold}"
                )

        if cfg.pipeline_mode in ("danmu_plus_asr", "asr_only") and cfg.asr_trigger_threshold > 0:
            asr_count = len(self._asr_buffer)
            if asr_count < cfg.asr_trigger_threshold:
                blockers.append(
                    f"asr_sentence_count={asr_count} < asr_trigger_threshold={cfg.asr_trigger_threshold}"
                )
        return blockers

    def _extract_llm_text(self, resp: object) -> str:
        if resp is None:
            return ""
        if isinstance(resp, str):
            return resp.strip()
        for key in ("completion_text", "completion", "text", "content"):
            value = getattr(resp, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _normalize_reply(self, text: str, max_chars: int) -> str:
        text = str(text or "").strip().replace("\r", " ").replace("\n", " ")
        text = text.strip("`").strip()
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1].strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    def _prune_old(self, context_window_seconds: int):
        now = time.time()
        danmaku_cutoff = now - max(1, context_window_seconds)
        self._context_danmaku_buffer = [
            item for item in self._context_danmaku_buffer if item.ts >= danmaku_cutoff
        ]
        seen_ttl = max(600, context_window_seconds * 6)
        seen_cutoff = now - seen_ttl
        self._seen = {k: ts for k, ts in self._seen.items() if ts >= seen_cutoff}
        asr_cutoff = now - max(1, context_window_seconds)
        self._context_asr_buffer = [
            seg
            for seg in self._context_asr_buffer
            if (seg.wall_ts_end <= 0) or ((seg.wall_ts_end or seg.ts_end) >= asr_cutoff)
        ]

    def _resolve_target_umo(self, cfg: WatchConfig) -> str:
        platform_ids = self._get_available_platform_ids()
        platform_id = self._pick_platform_id(cfg.target_platform_id, platform_ids)

        raw_umo = cfg.target_umo.strip()
        if raw_umo:
            parts = raw_umo.split(":")
            if len(parts) >= 3:
                source_platform_id = parts[0].strip()
                msg_type = self._normalize_message_type(parts[1].strip())
                session_id = ":".join(parts[2:]).strip()
                if not session_id:
                    return ""
                use_platform_id = source_platform_id
                if platform_ids and source_platform_id not in platform_ids:
                    use_platform_id = platform_id
                return f"{use_platform_id}:{msg_type}:{session_id}"
            guessed_id = raw_umo
            msg_type = self._normalize_message_type(cfg.target_type)
            return f"{platform_id}:{msg_type}:{guessed_id}"

        if not cfg.target_id:
            return ""
        msg_type = self._normalize_message_type(cfg.target_type)
        return f"{platform_id}:{msg_type}:{cfg.target_id}"

    def _normalize_message_type(self, target_type: str) -> str:
        t = str(target_type or "").strip().lower()
        if t in ("friend", "private", "privatemessage", "私聊"):
            return "FriendMessage"
        if t in ("group", "groupmessage", "群聊"):
            return "GroupMessage"
        if t == "friendmessage":
            return "FriendMessage"
        return "GroupMessage"

    def _pick_platform_id(self, configured: str, available_ids: list[str]) -> str:
        configured = str(configured or "").strip()
        if configured and (not available_ids or configured in available_ids):
            return configured
        if available_ids:
            if configured and configured not in available_ids:
                logger.warning(
                    f"[bili_watcher] target_platform_id={configured} 不存在，自动改用 {available_ids[0]}"
                )
            return available_ids[0]
        return configured or "default"

    def _get_available_platform_ids(self) -> list[str]:
        pm = getattr(self.context, "platform_manager", None)
        if not pm:
            return []

        insts = []
        try:
            if hasattr(pm, "get_insts"):
                insts = pm.get_insts()
            elif hasattr(pm, "insts"):
                raw = pm.insts
                if isinstance(raw, dict):
                    insts = list(raw.values())
                elif isinstance(raw, list):
                    insts = raw
        except Exception as e:
            logger.warning(f"[bili_watcher] get platform instances failed: {e}")
            return []

        ids: list[str] = []
        for inst in insts:
            pid = None
            if hasattr(inst, "metadata") and getattr(inst.metadata, "id", None):
                pid = inst.metadata.id
            elif hasattr(inst, "id"):
                pid = inst.id
            if pid:
                sid = str(pid).strip()
                if sid and sid not in ids:
                    ids.append(sid)
        return ids

    def _resolve_plugin_path(self, raw_path: str, default_path: str = "") -> str:
        candidate = str(raw_path or default_path or "").strip()
        if not candidate:
            return ""
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (PLUGIN_DIR / path).resolve()
        return str(path)

    def _normalize_pipeline_mode(self, raw_mode: object) -> str:
        mode = str(raw_mode or "danmu_only").strip().lower()
        if mode in {"danmu_only", "danmu_plus_asr", "asr_only"}:
            return mode
        return "danmu_only"

    def _normalize_asr_strategy(self, raw_strategy: object) -> str:
        strategy = str(raw_strategy or "sensevoice_vad_offline").strip().lower()
        aliases = {
            "streaming": "streaming_zipformer",
            "zipformer": "streaming_zipformer",
            "sensevoice": "sensevoice_vad_offline",
            "sense_voice": "sensevoice_vad_offline",
            "sensevoice_vad": "sensevoice_vad_offline",
        }
        strategy = aliases.get(strategy, strategy)
        if strategy in {"streaming_zipformer", "sensevoice_vad_offline"}:
            return strategy
        return "sensevoice_vad_offline"

    def _load_config(self) -> WatchConfig:
        reply_interval_seconds = self._to_int(
            self.config.get("reply_interval_seconds",15),
            15,
            1,
        )
        danmaku_trigger_threshold = self._to_int(
            self.config.get("danmaku_trigger_threshold", 10),
            20,
            1,
        )
        asr_trigger_threshold = self._to_int(
            self.config.get("asr_trigger_threshold", 10),
            1,
            0,
        )
        default_context_window_seconds = max(
            reply_interval_seconds,
            reply_interval_seconds * DEFAULT_CONTEXT_WINDOW_MULTIPLIER,
        )
        context_window_seconds = self._to_int(
            self.config.get("context_window_seconds", default_context_window_seconds),
            default_context_window_seconds,
            1,
        )

        cookie = str(
            self.config.get("bili_cookie", self.config.get("bilibili_cookie", "")) or ""
        ).strip()
        cookie_file = str(
            self.config.get(
                "bili_cookie_file",
                self.config.get("bilibili_cookie_file", DEFAULT_COOKIE_FILE),
            )
            or DEFAULT_COOKIE_FILE
        ).strip()
        auto_load_cookie = self._to_bool(
            self.config.get("auto_load_cookie_from_file", True),
            True,
        )
        if not cookie and auto_load_cookie:
            cookie = self._load_cookie_from_file(cookie_file)

        return WatchConfig(
            enabled=self._to_bool(self.config.get("enabled", True), True),
            debug=self._to_bool(self.config.get("debug", False), False),
            room_id=self._to_int(self.config.get("room_id", 0), 0, 0),
            reply_interval_seconds=reply_interval_seconds,
            context_window_seconds=context_window_seconds,
            danmaku_trigger_threshold=danmaku_trigger_threshold,
            asr_trigger_threshold=asr_trigger_threshold,
            target_umo=str(self.config.get("target_umo", "") or "").strip(),
            target_platform_id=str(self.config.get("target_platform_id", "default") or "default").strip(),
            target_type=str(self.config.get("target_type", "group") or "group").strip(),
            target_id=str(self.config.get("target_id", "") or "").strip(),
            max_reply_chars=self._to_int(self.config.get("max_reply_chars", 60), 60, 10),
            bilibili_cookie=cookie,
            bilibili_cookie_file=cookie_file,
            auto_load_cookie_from_file=auto_load_cookie,
            pipeline_mode=self._normalize_pipeline_mode(self.config.get("pipeline_mode", "danmu_only")),
            use_realtime_danmaku_ws=self._to_bool(self.config.get("use_realtime_danmaku_ws", True), True),
            danmu_ws_auth_mode=str(self.config.get("danmu_ws_auth_mode", "signed_wbi") or "signed_wbi").strip(),
            allow_buvid3_only=self._to_bool(self.config.get("allow_buvid3_only", True), True),
            wbi_sign_enabled=self._to_bool(self.config.get("wbi_sign_enabled", True), True),
            audio_enabled=self._to_bool(self.config.get("audio_enabled", True), True),
            audio_pull_protocol=str(
                self.config.get("audio_pull_protocol", "http_flv") or "http_flv"
            ).strip(),
            audio_pull_api_preference=str(
                self.config.get("audio_pull_api_preference", "getRoomPlayInfo") or "getRoomPlayInfo"
            ).strip(),
            audio_http_headers_enabled=self._to_bool(
                self.config.get("audio_http_headers_enabled", True),
                True,
            ),
            ffmpeg_path=str(self.config.get("ffmpeg_path", "ffmpeg") or "ffmpeg").strip(),
            audio_sample_rate=self._to_int(self.config.get("audio_sample_rate", 16000), 16000, 8000),
            asr_backend=str(self.config.get("asr_backend", "sherpa_onnx_rknn") or "sherpa_onnx_rknn").strip(),
            asr_strategy=self._normalize_asr_strategy(
                self.config.get("asr_strategy", "sensevoice_vad_offline")
            ),
            asr_model_dir=self._resolve_plugin_path(
                str(self.config.get("asr_model_dir", DEFAULT_ASR_MODEL_DIR) or DEFAULT_ASR_MODEL_DIR).strip(),
                DEFAULT_ASR_MODEL_DIR,
            ),
            asr_vad_model_path=self._resolve_plugin_path(
                str(
                    self.config.get("asr_vad_model_path", DEFAULT_ASR_VAD_MODEL_PATH)
                    or DEFAULT_ASR_VAD_MODEL_PATH
                ).strip(),
                DEFAULT_ASR_VAD_MODEL_PATH,
            ),
            asr_vad_threshold=self._to_float(self.config.get("asr_vad_threshold", 0.3), 0.3),
            asr_vad_min_silence_duration=self._to_float(
                self.config.get("asr_vad_min_silence_duration", 0.35),
                0.35,
            ),
            asr_vad_min_speech_duration=self._to_float(
                self.config.get("asr_vad_min_speech_duration", 0.25),
                0.25,
            ),
            asr_vad_max_speech_duration=self._to_float(
                self.config.get("asr_vad_max_speech_duration", 20.0),
                20.0,
            ),
            asr_sense_voice_language=str(
                self.config.get("asr_sense_voice_language", "auto") or "auto"
            ).strip(),
            asr_sense_voice_use_itn=self._to_bool(
                self.config.get("asr_sense_voice_use_itn", True),
                True,
            ),
            asr_runtime_probe_required=self._to_bool(
                self.config.get("asr_runtime_probe_required", True),
                True,
            ),
            asr_threads=self._to_int(self.config.get("asr_threads", 1), 1, -4),
            asr_vad_enabled=self._to_bool(self.config.get("asr_vad_enabled", False), False),
            asr_sentence_pause_seconds=self._to_float(
                self.config.get("asr_sentence_pause_seconds", 0.8),
                0.8,
            ),
            asr_sentence_min_chars=self._to_int(
                self.config.get("asr_sentence_min_chars", 2),
                2,
                1,
            ),
            singer_mode_enabled=self._to_bool(self.config.get("singer_mode_enabled", True), True),
            singer_mode_threshold=self._to_float(self.config.get("singer_mode_threshold", 0.3), 0.3),
        )

    def _load_cookie_from_file(self, cookie_file: str) -> str:
        path = Path(cookie_file).expanduser()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("cookie", "") or "").strip()
        except Exception as e:
            logger.warning(f"[bili_watcher] load cookie file failed: {e}")
            return ""

    def _to_int(self, value: object, default: int, min_value: int) -> int:
        try:
            iv = int(float(value))
            if iv < min_value:
                return min_value
            return iv
        except Exception:
            return default

    def _to_float(self, value: object, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _to_bool(self, value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ("1", "true", "yes", "on", "y"):
                return True
            if value in ("0", "false", "no", "off", "n"):
                return False
        return default

    def _log_warn_throttled(self, msg: str, interval_seconds: int = 60):
        now = time.time()
        if now - self._last_warn_ts >= interval_seconds:
            logger.warning(msg)
            self._last_warn_ts = now

    def _set_config_value(self, key: str, value: object):
        try:
            self.config[key] = value
        except Exception:
            pass

    def _save_config_if_possible(self) -> bool:
        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
                return True
            except Exception as e:
                logger.warning(f"[bili_watcher] save_config failed: {e}")
        return False

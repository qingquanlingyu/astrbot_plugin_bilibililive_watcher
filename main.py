from __future__ import annotations

import asyncio
import inspect
import json
import re
import tempfile
import time
from pathlib import Path

import aiohttp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.entities import ProviderRequest

try:  # pragma: no cover
    import qrcode
except Exception:  # pragma: no cover
    qrcode = None

try:  # pragma: no cover
    from .asr_sherpa import SherpaASRWorker, build_asr_worker_or_none
    from .audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from .bili_http import (
        BiliApiError,
        BiliHttpClient,
        BiliLoginAccount,
        BiliLoginRequiredError,
        DEFAULT_UA,
    )
    from .bili_ws import DanmakuRealtimeClient
    from .fusion import DEFAULT_SINGER_KEYWORDS, FusionEngine
    from .models import ASRSegment, ChannelSendState, DanmakuItem, LoginRuntimeState, WatchConfig
    from .prompting import (
        DEFAULT_FUSED_PROMPT_TEMPLATE,
        DEFAULT_SINGER_MODE_INSTRUCTION,
        build_fused_prompt,
    )
except ImportError:  # pragma: no cover
    from asr_sherpa import SherpaASRWorker, build_asr_worker_or_none
    from audio_pipe import AudioCaptureWorker, AudioRequestOptions
    from bili_http import (
        BiliApiError,
        BiliHttpClient,
        BiliLoginAccount,
        BiliLoginRequiredError,
        DEFAULT_UA,
    )
    from bili_ws import DanmakuRealtimeClient
    from fusion import DEFAULT_SINGER_KEYWORDS, FusionEngine
    from models import ASRSegment, ChannelSendState, DanmakuItem, LoginRuntimeState, WatchConfig
    from prompting import DEFAULT_FUSED_PROMPT_TEMPLATE, DEFAULT_SINGER_MODE_INSTRUCTION, build_fused_prompt

DEFAULT_ASR_MODEL_DIR = (
    "./models/sherpa/rknn/"
    "sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
)
DEFAULT_ASR_VAD_MODEL_PATH = "./models/vad/silero_vad.onnx"
DEFAULT_CONVERSATION_CONTEXT_LIMIT = 12
DEFAULT_CONTEXT_WINDOW_MULTIPLIER = 3
DEFAULT_PCM_GAP_LOG_SECONDS = 2.0
DEFAULT_PCM_HEARTBEAT_SECONDS = 5.0
DEFAULT_HISTORY_POLL_INTERVAL_WHEN_WS_FATAL_SECONDS = 20.0
DEFAULT_WS_FATAL_RETRY_COOLDOWN_SECONDS = 20.0
DEFAULT_WS_IDLE_HISTORY_FALLBACK_SECONDS = 15.0
INTERNAL_USE_REALTIME_DANMAKU_WS = True
INTERNAL_DANMU_WS_AUTH_MODE = "signed_wbi"
INTERNAL_ALLOW_BUVID3_ONLY = True
INTERNAL_WBI_SIGN_ENABLED = True
INTERNAL_AUDIO_PULL_API_PREFERENCE = "getRoomPlayInfo"
INTERNAL_AUDIO_HTTP_HEADERS_ENABLED = True
INTERNAL_ASR_SENSE_VOICE_USE_ITN = True
INTERNAL_ASR_RUNTIME_PROBE_REQUIRED = True
HIDDEN_LEGACY_CONFIG_KEYS = (
    "bili_cookie_file",
    "bilibili_cookie_file",
    "auto_load_cookie_from_file",
    "audio_enabled",
    "asr_strategy",
    "asr_vad_enabled",
    "asr_sentence_pause_seconds",
    "asr_sentence_min_chars",
    "singer_mode_threshold",
    "use_realtime_danmaku_ws",
    "danmu_ws_auth_mode",
    "allow_buvid3_only",
    "wbi_sign_enabled",
    "audio_pull_api_preference",
    "audio_http_headers_enabled",
    "asr_sense_voice_use_itn",
    "asr_runtime_probe_required",
)
PLUGIN_DIR = Path(__file__).resolve().parent

@register(
    "astrbot_plugin_bilibililive_watcher",
    "YourName",
    "监听B站直播弹幕热度并触发模型生成短弹幕发送到指定会话。",
    "1.2.0",
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
        self._ws_retry_after_ts = 0.0
        self._ws_last_message_ts = 0.0
        self._last_history_poll_ts = 0.0
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
        self._astrbot_send_state = ChannelSendState(channel="astrbot", enabled=True)
        self._bili_live_send_state = ChannelSendState(channel="bilibili_live", enabled=False)
        self._login_runtime = LoginRuntimeState()
        self._login_poll_task: asyncio.Task | None = None
        self._account_status_cache: BiliLoginAccount | None = None
        self._account_status_cache_ts = 0.0

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

        if self._login_poll_task:
            self._login_poll_task.cancel()
            await asyncio.gather(self._login_poll_task, return_exceptions=True)
            self._login_poll_task = None
        self._cleanup_login_qrcode_image()

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
            "/biliwatch help - 查看完整命令说明",
            "/biliwatch status - 查看当前运行状态、房间号、登录态和发送状态",
            "/biliwatch toggle [on|off] - 开启或关闭整个插件，不传参数时自动切换",
            "/biliwatch room <room_id> - 设置要监听的 B 站直播间号",
            "/biliwatch bind - 把当前会话绑定为 AstrBot 侧发送目标",
            "/biliwatch sync-live [on|off] - 开启或关闭同步发送到 B 站直播弹幕",
            "/biliwatch reply-interval <seconds> - 设置主循环和常规 history 轮询间隔",
            "/biliwatch context-window <seconds> - 设置保留给融合和 prompt 的上下文窗口",
            "/biliwatch danmaku-threshold <count> - 设置触发前至少需要多少条弹幕",
            "/biliwatch asr-threshold <count> - 设置触发前至少需要多少条 ASR 语句",
            "/biliwatch login - 发起 B 站二维码登录（实验性）",
            "/biliwatch login status - 查看二维码登录进度和当前账号状态",
            "/biliwatch logout - 清除插件内保存的二维码登录态",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.command("biliwatch status")
    async def biliwatch_status(self, event: AstrMessageEvent):
        cfg = self._load_config()
        self._astrbot_send_state.enabled = True
        self._bili_live_send_state.enabled = cfg.sync_to_bilibili_live
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
        account_status = await self._get_bili_account_status(cfg, refresh=False)
        yield event.plain_result(
            "\n".join(
                [
                    f"enabled: {cfg.enabled}",
                    f"debug: {cfg.debug}",
                    f"pipeline_mode: {cfg.pipeline_mode}",
                    f"room_id: {cfg.room_id}",
                    f"generation_provider_id: {cfg.generation_provider_id or '(default)'}",
                    f"generation_persona_id: {cfg.generation_persona_id or '(default)'}",
                    (
                        "generation_prompt_template: default"
                        if cfg.generation_prompt_template.strip() == DEFAULT_FUSED_PROMPT_TEMPLATE.strip()
                        or not cfg.generation_prompt_template.strip()
                        else "generation_prompt_template: set"
                    ),
                    f"target: {target_umo or '(未配置)'}",
                    f"platform_ids: {platform_ids or '[]'}",
                    f"bili_cookie_source: {cfg.bilibili_cookie_source}",
                    f"bili_live_sync_enabled: {cfg.sync_to_bilibili_live}",
                    f"bili_login_status: {self._format_account_status(account_status)}",
                    f"bili_login_runtime: {self._format_login_runtime_status()}",
                    f"astrbot_send: {self._format_channel_send_state(self._astrbot_send_state)}",
                    f"bili_live_send: {self._format_channel_send_state(self._bili_live_send_state)}",
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

        self._set_config_value("global.room_id", room_id_int)
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

        self._set_config_value("sender.target_umo", umo)
        parts = umo.split(":")
        if len(parts) >= 3:
            self._set_config_value("sender.target_platform_id", parts[0].strip())
            msg_type = parts[1].strip().lower()
            if "friend" in msg_type or "private" in msg_type:
                self._set_config_value("sender.target_type", "private")
            else:
                self._set_config_value("sender.target_type", "group")
            self._set_config_value("sender.target_id", ":".join(parts[2:]).strip())

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

        current = self._to_bool(self._config_get("global.enabled", True, legacy_keys=("enabled",)), True)
        if raw in ("on", "enable", "enabled", "true", "1", "开", "开启"):
            new_value = True
        elif raw in ("off", "disable", "disabled", "false", "0", "关", "关闭"):
            new_value = False
        else:
            new_value = not current

        self._set_config_value("global.enabled", new_value)
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        state = "开启" if new_value else "关闭"
        yield event.plain_result(f"已{state} B站直播监听 {suffix}")

    @filter.command("biliwatch sync-live")
    async def biliwatch_sync_live(self, event: AstrMessageEvent, value: str = ""):
        raw = str(value or "").strip().lower()
        if not raw:
            parts = event.message_str.strip().split()
            if len(parts) >= 3:
                raw = parts[-1].strip().lower()

        current = self._to_bool(
            self._config_get("global.sync_to_bilibili_live", False, legacy_keys=("sync_to_bilibili_live",)),
            False,
        )
        if raw in ("on", "enable", "enabled", "true", "1", "开", "开启"):
            new_value = True
        elif raw in ("off", "disable", "disabled", "false", "0", "关", "关闭"):
            new_value = False
        else:
            new_value = not current

        self._set_config_value("global.sync_to_bilibili_live", new_value)
        self._bili_live_send_state.enabled = new_value
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        state = "开启" if new_value else "关闭"
        yield event.plain_result(f"已{state}同步发送到 B 站直播弹幕 {suffix}")

    @filter.command("biliwatch reply-interval")
    async def biliwatch_set_reply_interval(self, event: AstrMessageEvent, value: str = ""):
        result = self._set_integer_config_from_command(
            event,
            explicit_value=value,
            config_key="main_loop.reply_interval_seconds",
            label="reply_interval_seconds",
            min_value=1,
            example="/biliwatch reply-interval 15",
        )
        yield event.plain_result(result)

    @filter.command("biliwatch context-window")
    async def biliwatch_set_context_window(self, event: AstrMessageEvent, value: str = ""):
        result = self._set_integer_config_from_command(
            event,
            explicit_value=value,
            config_key="main_loop.context_window_seconds",
            label="context_window_seconds",
            min_value=1,
            example="/biliwatch context-window 45",
        )
        yield event.plain_result(result)

    @filter.command("biliwatch danmaku-threshold")
    async def biliwatch_set_danmaku_threshold(self, event: AstrMessageEvent, value: str = ""):
        result = self._set_integer_config_from_command(
            event,
            explicit_value=value,
            config_key="main_loop.danmaku_trigger_threshold",
            label="danmaku_trigger_threshold",
            min_value=0,
            example="/biliwatch danmaku-threshold 10",
        )
        yield event.plain_result(result)

    @filter.command("biliwatch asr-threshold")
    async def biliwatch_set_asr_threshold(self, event: AstrMessageEvent, value: str = ""):
        result = self._set_integer_config_from_command(
            event,
            explicit_value=value,
            config_key="main_loop.asr_trigger_threshold",
            label="asr_trigger_threshold",
            min_value=0,
            example="/biliwatch asr-threshold 2",
        )
        yield event.plain_result(result)

    @filter.command("biliwatch login")
    async def biliwatch_login(self, event: AstrMessageEvent, action: str = ""):
        raw = str(action or "").strip().lower()
        if not raw:
            parts = event.message_str.strip().split()
            if len(parts) >= 3:
                raw = parts[2].strip().lower()

        if raw == "status":
            cfg = self._load_config()
            account_status = await self._get_bili_account_status(cfg, refresh=True)
            yield event.plain_result(
                "\n".join(
                    [
                        f"login_runtime: {self._format_login_runtime_status()}",
                        f"bili_login_status: {self._format_account_status(account_status)}",
                        "说明：二维码登录为实验性能力，如失败可回退到 bili_cookie。",
                    ]
                )
            )
            return

        if self._http is None:
            yield event.plain_result("HTTP 客户端未初始化，请稍后重试。")
            return

        if self._login_poll_task and not self._login_poll_task.done():
            seconds_left = max(0, int(self._login_runtime.expires_at - time.time()))
            yield event.plain_result(
                "\n".join(
                    [
                        "已有二维码登录流程正在进行中。",
                        f"status: {self._login_runtime.status}",
                        f"url: {self._login_runtime.url or '(空)'}",
                        f"expires_in_seconds: {seconds_left}",
                        "可用 /biliwatch login status 查看最新进度。",
                    ]
                )
            )
            return

        try:
            qr_session = await self._http.generate_login_qrcode()
        except Exception as e:
            yield event.plain_result(
                "发起二维码登录失败："
                f"{self._sanitize_error_message(e)}\n"
                "说明：二维码登录为实验性能力，可回退到 bili_cookie。"
            )
            return

        self._cleanup_login_qrcode_image()
        qr_image_path = await self._build_login_qrcode_image(qr_session.url)
        self._login_runtime = LoginRuntimeState(
            qrcode_key=qr_session.qrcode_key,
            url=qr_session.url,
            image_path=qr_image_path or "",
            status="waiting_scan",
            message="waiting_scan",
            started_ts=time.time(),
            expires_at=time.time() + max(30, qr_session.expires_in_seconds),
            completed_ts=0.0,
        )
        self._login_poll_task = asyncio.create_task(
            self._poll_login_until_complete(),
            name="bili-login-poll",
        )
        if qr_image_path:
            try:
                yield event.image_result(qr_image_path)
            except Exception as e:
                logger.warning("[bili_watcher] send login qrcode image failed: %s", e)
        yield event.plain_result(
            "\n".join(
                [
                    "已发起 B 站二维码登录（实验性）。",
                    f"qrcode_image: {'sent' if qr_image_path else 'unavailable'}",
                    f"login_url: {qr_session.url}",
                    f"qrcode_key: {self._mask_qrcode_key(qr_session.qrcode_key)}",
                    "说明：若上方图片未显示，请直接打开 login_url。",
                    "请在扫码后使用 /biliwatch login status 查看确认结果。",
                ]
            )
        )

    @filter.command("biliwatch logout")
    async def biliwatch_logout(self, event: AstrMessageEvent):
        await self._clear_persisted_login_state(clear_manual_cookie=False)
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        cfg = self._load_config()
        fallback_hint = ""
        if cfg.bilibili_cookie_source == "config":
            fallback_hint = "\n注意：当前仍存在手工 bili_cookie 回退来源，B 站发送仍可能继续可用。"
        yield event.plain_result(f"已清除插件内保存的 B 站登录态 {suffix}{fallback_hint}")

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
            runtime_alive = bool(
                (self._audio_task and not self._audio_task.done())
                or (self._ws_client is not None and self._ws_client.running)
            )
            if runtime_alive and self._runtime_room_id:
                self._log_warn_throttled(
                    "[bili_watcher] 本轮配置读取到 room_id<=0，已跳过新一轮轮询；"
                    f"但已有 room_id={self._runtime_room_id} 的运行时仍在继续"
                )
            else:
                self._log_warn_throttled("[bili_watcher] room_id 未配置，已跳过轮询")
            return

        target_umo = self._resolve_target_umo(cfg)
        if not target_umo:
            self._log_warn_throttled("[bili_watcher] 发送目标未配置，已跳过轮询")
            return

        room_id = await self._resolve_real_room_id(cfg.room_id, cfg.bilibili_cookie)
        await self._ensure_runtime(cfg, room_id)

        if self._should_poll_history(cfg):
            self._mark_history_poll_attempt()
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

        await self._dispatch_reply(
            cfg=cfg,
            target_umo=target_umo,
            room_id=room_id,
            reply=reply,
        )

    async def _ensure_runtime(self, cfg: WatchConfig, room_id: int):
        if self._runtime_room_id != room_id:
            await self._stop_runtime_clients()
            self._runtime_room_id = room_id
            self._ws_retry_after_ts = 0.0
            self._ws_last_message_ts = 0.0
            self._last_history_poll_ts = 0.0
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
        if self._should_delay_ws_retry(ws_key):
            return

        await self._stop_ws_runtime()
        self._ws_client = DanmakuRealtimeClient(
            http_client=self._http,
            room_id=room_id,
            cookie=cfg.bilibili_cookie,
            wbi_cookie=wbi_cookie,
            ws_require_wbi_sign=cfg.wbi_sign_enabled,
            prefer_buvid3_ws_cookie=cfg.allow_buvid3_only,
            on_danmaku=self._on_realtime_danmaku,
        )
        self._ws_runtime_key = ws_key
        await self._ws_client.start()

    async def _ensure_audio_runtime(self, cfg: WatchConfig, room_id: int):
        audio_enabled = cfg.pipeline_mode in ("danmu_plus_asr", "asr_only")
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
        self._ws_retry_after_ts = 0.0
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
            )
        worker = SherpaASRWorker(
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
                await capture.run(
                    urls[0],
                    self._on_pcm,
                    request_options=request_options,
                    on_stderr=self._on_ffmpeg_stderr,
                )
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

    async def _on_ffmpeg_stderr(self, line: str):
        text = self._sanitize_error_message(line)
        if text and text != "unknown_error":
            logger.warning("[bili_watcher] ffmpeg stderr: %s", text)

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
        reason = self._history_poll_reason(cfg)
        if reason is None:
            return False
        if self._last_history_poll_ts <= 0:
            return True
        interval_seconds = self._history_poll_interval_seconds(cfg, reason)
        return (time.time() - self._last_history_poll_ts) >= interval_seconds

    def _history_poll_reason(self, cfg: WatchConfig) -> str | None:
        if cfg.pipeline_mode == "asr_only":
            return None
        if not cfg.use_realtime_danmaku_ws or cfg.danmu_ws_auth_mode == "history_only":
            return "history_only"
        if self._ws_client is None:
            return "ws_unavailable"
        if self._ws_client.fatal_error:
            return "ws_fatal"
        if not self._ws_client.connected:
            return "ws_disconnected"
        idle_threshold = DEFAULT_WS_IDLE_HISTORY_FALLBACK_SECONDS
        if self._ws_last_message_ts <= 0:
            connected_at = float(getattr(self._ws_client, "connected_at", 0.0) or 0.0)
            if connected_at > 0 and (time.time() - connected_at) < idle_threshold:
                return None
            return "ws_no_message_yet"
        if (time.time() - self._ws_last_message_ts) >= idle_threshold:
            return "ws_idle"
        return None

    def _history_poll_interval_seconds(self, cfg: WatchConfig, reason: str) -> float:
        interval_seconds = float(max(1, cfg.reply_interval_seconds))
        if reason == "ws_fatal":
            return max(interval_seconds, DEFAULT_HISTORY_POLL_INTERVAL_WHEN_WS_FATAL_SECONDS)
        return interval_seconds

    def _mark_history_poll_attempt(self) -> None:
        self._last_history_poll_ts = time.time()

    def _should_delay_ws_retry(self, ws_key: tuple) -> bool:
        if self._ws_client is None or self._ws_runtime_key != ws_key:
            self._ws_retry_after_ts = 0.0
            return False
        if not str(self._ws_client.fatal_error or "").strip():
            self._ws_retry_after_ts = 0.0
            return False
        now = time.time()
        if self._ws_retry_after_ts <= 0:
            self._ws_retry_after_ts = now + DEFAULT_WS_FATAL_RETRY_COOLDOWN_SECONDS
        if now < self._ws_retry_after_ts:
            return True
        self._ws_retry_after_ts = 0.0
        return False

    async def _generate_short_reply(
        self,
        cfg: WatchConfig,
        target_umo: str,
        room_id: int,
        danmaku_items: list[DanmakuItem],
        asr_segments: list[ASRSegment],
    ) -> str:
        provider = await self._resolve_provider(cfg=cfg, target_umo=target_umo)
        if provider is None:
            logger.warning("[bili_watcher] no provider available")
            return ""

        conversation, contexts = await self._get_recent_contexts(
            target_umo=target_umo,
            max_messages=DEFAULT_CONVERSATION_CONTEXT_LIMIT,
        )
        system_prompt = await self._get_system_prompt(cfg=cfg, umo=target_umo, conversation=conversation)
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
            singer_mode_keywords=cfg.singer_mode_keywords,
            singer_mode_window_seconds=cfg.singer_mode_window_seconds,
        )
        fusion.ordered_context = self._build_ordered_context(
            danmaku_items=danmaku_items,
            asr_segments=asr_segments,
        )
        account_status = await self._get_bili_account_status(cfg, refresh=False)
        prompt = build_fused_prompt(
            room_id=room_id,
            room_title=room_meta.get("room_title", ""),
            anchor_name=room_meta.get("anchor_name", ""),
            self_bili_nickname=str(getattr(account_status, "uname", "") or cfg.bili_login_uname or "").strip(),
            fusion=fusion,
            max_reply_chars=cfg.max_reply_chars,
            prompt_template=cfg.generation_prompt_template,
            singer_mode_instruction=cfg.singer_mode_instruction,
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

    async def _dispatch_reply(
        self,
        *,
        cfg: WatchConfig,
        target_umo: str,
        room_id: int,
        reply: str,
    ) -> None:
        self._astrbot_send_state.enabled = True
        self._bili_live_send_state.enabled = cfg.sync_to_bilibili_live
        await self._send_astrbot_reply(target_umo=target_umo, room_id=room_id, reply=reply)
        if cfg.sync_to_bilibili_live:
            await self._send_bili_live_reply(cfg=cfg, room_id=room_id, reply=reply)

    async def _send_astrbot_reply(self, *, target_umo: str, room_id: int, reply: str) -> None:
        try:
            ok = await self.context.send_message(target_umo, MessageChain().message(reply))
        except Exception as e:
            self._record_channel_send_result(
                self._astrbot_send_state,
                ok=False,
                summary="exception",
                error=self._sanitize_error_message(e),
                text_preview=reply,
            )
            logger.warning("[bili_watcher] send_message exception, target=%s err=%s", target_umo, e)
            return

        if ok:
            self._record_channel_send_result(
                self._astrbot_send_state,
                ok=True,
                summary=f"sent target={target_umo}",
                error="",
                text_preview=reply,
            )
            logger.info(f"[bili_watcher] sent to {target_umo}, room={room_id}, text={reply[:120]}")
            return

        self._record_channel_send_result(
            self._astrbot_send_state,
            ok=False,
            summary=f"failed target={target_umo}",
            error="send_message returned false",
            text_preview=reply,
        )
        logger.warning(f"[bili_watcher] send_message failed, target={target_umo}")

    async def _send_bili_live_reply(self, *, cfg: WatchConfig, room_id: int, reply: str) -> None:
        if self._http is None:
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=False,
                summary="http_client_unavailable",
                error="BiliHttpClient is not initialized",
                text_preview=reply,
            )
            return

        if not cfg.bilibili_cookie:
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=False,
                summary="not_logged_in",
                error="no effective bilibili cookie available",
                text_preview=reply,
            )
            return

        try:
            result = await self._http.send_live_danmaku(
                room_id=room_id,
                message=reply,
                cookie=cfg.bilibili_cookie,
            )
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=True,
                summary=f"sent code={result.code} msg={result.message or 'ok'}",
                error="",
                text_preview=reply,
            )
            logger.info(
                "[bili_watcher] sent live danmaku room=%s source=%s text=%s",
                room_id,
                cfg.bilibili_cookie_source,
                reply[:120],
            )
            return
        except BiliLoginRequiredError as e:
            self._account_status_cache = BiliLoginAccount(
                is_logged_in=False,
                source=cfg.bilibili_cookie_source,
                message="login_required",
            )
            self._account_status_cache_ts = time.time()
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=False,
                summary="login_required",
                error=self._sanitize_error_message(e),
                text_preview=reply,
            )
        except BiliApiError as e:
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=False,
                summary="api_error",
                error=self._sanitize_error_message(e),
                text_preview=reply,
            )
        except Exception as e:
            self._record_channel_send_result(
                self._bili_live_send_state,
                ok=False,
                summary="exception",
                error=self._sanitize_error_message(e),
                text_preview=reply,
            )
        logger.warning(
            "[bili_watcher] live danmaku send failed room=%s source=%s summary=%s error=%s",
            room_id,
            cfg.bilibili_cookie_source,
            self._bili_live_send_state.summary,
            self._bili_live_send_state.error,
        )

    def _record_channel_send_result(
        self,
        state: ChannelSendState,
        *,
        ok: bool,
        summary: str,
        error: str,
        text_preview: str,
    ) -> None:
        now = time.time()
        state.last_attempt_ts = now
        state.ok = bool(ok)
        state.summary = str(summary or "").strip() or ("ok" if ok else "failed")
        state.error = str(error or "").strip()
        state.text_preview = self._mask_reply_preview(text_preview)
        if ok:
            state.last_success_ts = now

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
            if normalized in {"直播中", "未开播"}:
                return normalized
            lowered = normalized.lower()
            if lowered in {"1", "live", "on", "streaming"}:
                return "直播中"
            else:
                return "未开播"
        if raw_status is None:
            return "未开播"
        try:
            value = int(raw_status)
        except (TypeError, ValueError):
            return "未开播"
        if value == 1:
            return "直播中"
        else:
            return "未开播"

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

    async def _maybe_await(self, value: object) -> object:
        if inspect.isawaitable(value):
            return await value
        return value

    def _get_runtime_config(self, *, umo: str) -> dict:
        getter = getattr(self.context, "get_config", None)
        if not callable(getter):
            return {}
        try:
            cfg = getter(umo=umo)
        except TypeError:
            try:
                cfg = getter()
            except Exception:
                return {}
        except Exception as e:
            logger.warning("[bili_watcher] get runtime config failed umo=%s err=%s", umo, e)
            return {}
        if isinstance(cfg, dict):
            return cfg
        return {}

    def _get_runtime_provider_settings(self, *, umo: str) -> dict:
        runtime_cfg = self._get_runtime_config(umo=umo)
        provider_settings = runtime_cfg.get("provider_settings")
        if isinstance(provider_settings, dict):
            return provider_settings
        return {}

    def _get_runtime_provider_id(self, *, umo: str) -> str:
        runtime_cfg = self._get_runtime_config(umo=umo)
        return str(runtime_cfg.get("provider_id", "") or "").strip()

    async def _resolve_provider(self, *, cfg: WatchConfig, target_umo: str):
        if cfg.generation_provider_id:
            provider = await self._resolve_provider_by_id(cfg.generation_provider_id)
            if provider is not None:
                return provider
            logger.warning(
                "[bili_watcher] configured provider not found: %s; fallback to current provider",
                cfg.generation_provider_id,
            )
        runtime_provider_id = self._get_runtime_provider_id(umo=target_umo)
        if runtime_provider_id:
            provider = await self._resolve_provider_by_id(runtime_provider_id)
            if provider is not None:
                return provider
        try:
            provider = self.context.get_using_provider(umo=target_umo)
        except TypeError:
            provider = self.context.get_using_provider()
        if not provider:
            provider = self.context.get_using_provider()
        return provider

    async def _resolve_provider_by_id(self, provider_id: str):
        target = str(provider_id or "").strip()
        if not target:
            return None
        manager = getattr(self.context, "provider_manager", None)
        for owner in (manager, self.context):
            if owner is None:
                continue
            for attr in ("get_provider", "get_provider_by_id", "find_provider", "get"):
                fn = getattr(owner, attr, None)
                if not callable(fn):
                    continue
                try:
                    result = await self._maybe_await(fn(target))
                except TypeError:
                    continue
                except Exception:
                    continue
                if result is not None:
                    return result
        for owner in (manager, self.context):
            provider = self._find_named_object_in_container(owner, target)
            if provider is not None:
                return provider
        return None

    def _find_named_object_in_container(self, owner: object, target: str):
        if owner is None:
            return None
        candidate_attrs = (
            "providers",
            "_providers",
            "provider_insts",
            "provider_instances",
            "all_providers",
            "personas",
            "_personas",
            "persona_insts",
            "persona_instances",
            "all_personas",
        )
        for attr in candidate_attrs:
            container = getattr(owner, attr, None)
            if container is None:
                continue
            if isinstance(container, dict):
                for key, item in container.items():
                    if str(key or "").strip() == target or self._object_matches_id(item, target):
                        return item
                continue
            items = container
            try:
                for item in items:
                    if self._object_matches_id(item, target):
                        return item
            except Exception:
                continue
        return None

    def _object_matches_id(self, item: object, target: str) -> bool:
        if isinstance(item, dict):
            values = [
                item.get("id"),
                item.get("provider_id"),
                item.get("persona_id"),
                item.get("name"),
            ]
            return any(str(value or "").strip() == target for value in values)
        values = [
            getattr(item, "id", None),
            getattr(item, "provider_id", None),
            getattr(item, "persona_id", None),
            getattr(getattr(item, "metadata", None), "id", None),
            getattr(getattr(item, "meta", None), "id", None),
            getattr(item, "name", None),
        ]
        for value in values:
            if str(value or "").strip() == target:
                return True
        return False

    def _extract_persona_prompt_text(self, persona: object) -> str:
        if persona is None:
            return ""
        if isinstance(persona, dict):
            return str(persona.get("system_prompt", "") or persona.get("prompt", "")).strip()
        for attr in ("system_prompt", "prompt"):
            value = getattr(persona, attr, None)
            text = str(value or "").strip()
            if text:
                return text
        return ""

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
            if self._is_invalid_system_role_error(e):
                llm_resp = await self._call_provider_without_system_roles(
                    provider=provider,
                    prompt=prompt,
                    contexts=contexts,
                    system_prompt=system_prompt,
                )
                if llm_resp is None:
                    logger.warning(f"[bili_watcher] provider.text_chat failed after system-role fallback: {e}")
            elif self._is_tool_context_mismatch_error(e):
                llm_resp = await self._call_provider_without_contexts(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                if llm_resp is None:
                    logger.warning(f"[bili_watcher] provider.text_chat failed after tool-context fallback: {e}")
            else:
                logger.warning(f"[bili_watcher] provider.text_chat failed: {e}")
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

    async def _call_provider_without_system_roles(
        self,
        *,
        provider: object,
        prompt: str,
        contexts: list[dict],
        system_prompt: str,
    ) -> object | None:
        safe_prompt = self._merge_prompt_with_system_instructions(
            prompt=prompt,
            contexts=contexts,
            system_prompt=system_prompt,
        )
        safe_contexts = self._strip_system_contexts(contexts)
        try:
            return await provider.text_chat(prompt=safe_prompt, contexts=safe_contexts)
        except TypeError:
            pass
        except Exception as e:
            logger.warning(f"[bili_watcher] retry without system roles failed: {e}")
            if self._is_tool_context_mismatch_error(e):
                return await self._call_provider_without_contexts(provider=provider, prompt=safe_prompt, system_prompt="")
            return None
        try:
            return await provider.text_chat(prompt=safe_prompt)
        except TypeError:
            try:
                return await provider.text_chat(safe_prompt)
            except Exception as e:
                logger.warning(f"[bili_watcher] retry without system roles/pure prompt positional failed: {e}")
                return None
        except Exception as e:
            logger.warning(f"[bili_watcher] retry without system roles/pure prompt failed: {e}")
            return None

    def _is_tool_context_mismatch_error(self, err: Exception) -> bool:
        text = str(err).lower()
        keywords = ("tool id() not found", "tool result's tool id", "tool_call_id", "tool result")
        return any(k in text for k in keywords)

    def _is_invalid_system_role_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return "invalid message role: system" in text or "invalid role: system" in text

    async def _get_system_prompt(self, *, cfg: WatchConfig, umo: str, conversation: object | None) -> str:
        persona_mgr = getattr(self.context, "persona_manager", None)
        if not persona_mgr:
            return ""
        try:
            if cfg.generation_persona_id:
                persona = await self._resolve_persona_by_id(persona_mgr, cfg.generation_persona_id)
                prompt = self._extract_persona_prompt_text(persona)
                if prompt:
                    return prompt
            selected_persona_prompt = await self._get_selected_persona_prompt(
                persona_mgr=persona_mgr,
                umo=umo,
                conversation=conversation,
            )
            if selected_persona_prompt:
                return selected_persona_prompt
            if conversation and getattr(conversation, "persona_id", None):
                persona = await persona_mgr.get_persona(conversation.persona_id)
                prompt = self._extract_persona_prompt_text(persona)
                if prompt:
                    return prompt
            try:
                default_persona = await persona_mgr.get_default_persona_v3(umo=umo)
            except TypeError:
                default_persona = await persona_mgr.get_default_persona_v3()
            return self._extract_persona_prompt_text(default_persona)
        except Exception as e:
            logger.warning(f"[bili_watcher] get system prompt failed: {e}")
        return ""

    async def _get_selected_persona_prompt(
        self,
        *,
        persona_mgr: object,
        umo: str,
        conversation: object | None,
    ) -> str:
        resolve_selected_persona = getattr(persona_mgr, "resolve_selected_persona", None)
        if not callable(resolve_selected_persona):
            return ""
        conversation_persona_id = None
        if conversation is not None:
            raw_persona_id = getattr(conversation, "persona_id", None)
            if raw_persona_id is not None:
                text = str(raw_persona_id).strip()
                conversation_persona_id = text if text else None
        try:
            try:
                resolved = await self._maybe_await(
                    resolve_selected_persona(
                        umo=umo,
                        conversation_persona_id=conversation_persona_id,
                        platform_name="",
                        provider_settings=self._get_runtime_provider_settings(umo=umo),
                    )
                )
            except TypeError:
                resolved = await self._maybe_await(
                    resolve_selected_persona(
                        umo=umo,
                        conversation_persona_id=conversation_persona_id,
                    )
                )
            selected_persona_id, selected_persona, _, _ = resolved
        except Exception as e:
            logger.warning("[bili_watcher] resolve selected persona failed umo=%s err=%s", umo, e)
            return ""
        prompt = self._extract_persona_prompt_text(selected_persona)
        if prompt:
            return prompt
        if selected_persona_id:
            persona = await self._resolve_persona_by_id(persona_mgr, str(selected_persona_id))
            return self._extract_persona_prompt_text(persona)
        return ""

    async def _resolve_persona_by_id(self, persona_mgr: object, persona_id: str):
        target = str(persona_id or "").strip()
        if not target:
            return None
        for attr in ("get_persona", "get_persona_by_id", "find_persona", "get"):
            fn = getattr(persona_mgr, attr, None)
            if not callable(fn):
                continue
            try:
                result = await self._maybe_await(fn(target))
            except TypeError:
                continue
            except Exception:
                continue
            if result is not None:
                return result
        return self._find_named_object_in_container(persona_mgr, target)

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

    @staticmethod
    def _strip_system_contexts(contexts: list[dict]) -> list[dict]:
        safe_contexts: list[dict] = []
        for item in contexts or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip()
            content = str(item.get("content", "") or "").strip()
            if role in ("user", "assistant") and content:
                safe_contexts.append({"role": role, "content": content})
        return safe_contexts

    @staticmethod
    def _merge_prompt_with_system_instructions(
        *,
        prompt: str,
        contexts: list[dict],
        system_prompt: str,
    ) -> str:
        instructions: list[str] = []
        system_text = str(system_prompt or "").strip()
        if system_text:
            instructions.append(system_text)
        for item in contexts or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "") or "").strip() != "system":
                continue
            content = str(item.get("content", "") or "").strip()
            if content:
                instructions.append(content)
        merged_prompt = str(prompt or "").strip()
        if not instructions:
            return merged_prompt
        instruction_block = "\n\n".join(instructions)
        if merged_prompt:
            return (
                "请先严格遵循以下附加指令，再生成回复：\n"
                f"{instruction_block}\n\n"
                "以下是本次需要回复的用户任务：\n"
                f"{merged_prompt}"
            )
        return instruction_block

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
        aliases = {
            0: "danmu_only",
            1: "asr_only",
            2: "danmu_plus_asr",
            "0": "danmu_only",
            "1": "asr_only",
            "2": "danmu_plus_asr",
            "danmu_only": "danmu_only",
            "asr_only": "asr_only",
            "danmu_plus_asr": "danmu_plus_asr",
        }
        if isinstance(raw_mode, str):
            mode = raw_mode.strip().lower()
            return aliases.get(mode, "danmu_only")
        if isinstance(raw_mode, (int, float)):
            try:
                return aliases.get(int(raw_mode), "danmu_only")
            except Exception:
                return "danmu_only"
        return "danmu_only"

    async def _poll_login_until_complete(self) -> None:
        try:
            while True:
                if self._http is None:
                    self._login_runtime.status = "error"
                    self._login_runtime.message = "http_client_unavailable"
                    self._login_runtime.completed_ts = time.time()
                    return
                if not self._login_runtime.qrcode_key:
                    self._login_runtime.status = "error"
                    self._login_runtime.message = "missing_qrcode_key"
                    self._login_runtime.completed_ts = time.time()
                    return
                if self._login_runtime.expires_at > 0 and time.time() >= self._login_runtime.expires_at:
                    self._login_runtime.status = "expired"
                    self._login_runtime.message = "二维码已过期"
                    self._login_runtime.completed_ts = time.time()
                    return

                result = await self._http.poll_login_qrcode(self._login_runtime.qrcode_key)
                self._login_runtime.status = result.status
                self._login_runtime.message = result.message
                if result.status in {"waiting_scan", "waiting_confirm"}:
                    await asyncio.sleep(2.5)
                    continue
                if result.status == "confirmed":
                    await self._persist_login_success(
                        cookie=result.cookie,
                        refresh_token=result.refresh_token,
                        account=result.account,
                    )
                    self._login_runtime.status = "confirmed"
                    self._login_runtime.message = "登录成功"
                    self._login_runtime.completed_ts = time.time()
                    return
                self._login_runtime.completed_ts = time.time()
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._login_runtime.status = "error"
            self._login_runtime.message = self._sanitize_error_message(e)
            self._login_runtime.completed_ts = time.time()
            logger.warning("[bili_watcher] qrcode login poll failed: %s", e)

    async def _persist_login_success(
        self,
        *,
        cookie: str,
        refresh_token: str,
        account: BiliLoginAccount | None,
    ) -> None:
        if not cookie:
            raise RuntimeError("qrcode login succeeded but returned empty cookie")

        resolved_account = account
        if (resolved_account is None or not resolved_account.is_logged_in) and self._http is not None:
            try:
                resolved_account = await self._http.get_login_account(cookie)
            except Exception:
                resolved_account = account

        existing_manual_cookie = str(
            self._config_get("user_auth.bili_cookie", "", legacy_keys=("bili_cookie", "bilibili_cookie")) or ""
        ).strip()
        self._set_config_value("user_auth.bili_login_cookie", cookie)
        self._set_config_value("user_auth.bili_login_refresh_token", refresh_token)
        self._set_config_value(
            "user_auth.bili_login_uid",
            str(getattr(resolved_account, "uid", "") or "").strip(),
        )
        self._set_config_value(
            "user_auth.bili_login_uname",
            str(getattr(resolved_account, "uname", "") or "").strip(),
        )
        self._set_config_value("user_auth.bili_login_saved_at", int(time.time()))
        if not existing_manual_cookie:
            self._set_config_value("user_auth.bili_cookie", cookie)
        self._save_config_if_possible()

        if resolved_account is not None:
            self._account_status_cache = resolved_account
            self._account_status_cache_ts = time.time()

    async def _clear_persisted_login_state(self, *, clear_manual_cookie: bool) -> None:
        current_task = asyncio.current_task()
        if self._login_poll_task and self._login_poll_task is not current_task:
            self._login_poll_task.cancel()
            await asyncio.gather(self._login_poll_task, return_exceptions=True)
        self._login_poll_task = None
        self._cleanup_login_qrcode_image()
        self._login_runtime = LoginRuntimeState()
        self._account_status_cache = None
        self._account_status_cache_ts = 0.0

        self._set_config_value("user_auth.bili_login_cookie", "")
        self._set_config_value("user_auth.bili_login_refresh_token", "")
        self._set_config_value("user_auth.bili_login_uid", "")
        self._set_config_value("user_auth.bili_login_uname", "")
        self._set_config_value("user_auth.bili_login_saved_at", 0)
        if clear_manual_cookie:
            self._set_config_value("user_auth.bili_cookie", "")

    async def _get_bili_account_status(
        self,
        cfg: WatchConfig,
        *,
        refresh: bool,
    ) -> BiliLoginAccount:
        if not cfg.bilibili_cookie:
            return self._saved_or_default_account_status(cfg, message="no_cookie")

        if (
            not refresh
            and self._account_status_cache is not None
            and (time.time() - self._account_status_cache_ts) < 60
        ):
            return self._account_status_cache

        if self._http is None:
            return self._saved_or_default_account_status(cfg, message="http_client_unavailable")

        try:
            account = await self._http.get_login_account(cfg.bilibili_cookie)
        except Exception as e:
            return self._saved_or_default_account_status(
                cfg,
                message=f"status_probe_failed:{self._sanitize_error_message(e)}",
            )

        if account.is_logged_in and not account.uname and cfg.bili_login_uname:
            account.uname = cfg.bili_login_uname
        if account.is_logged_in and not account.uid and cfg.bili_login_uid:
            account.uid = cfg.bili_login_uid
        if not account.source:
            account.source = cfg.bilibili_cookie_source
        self._account_status_cache = account
        self._account_status_cache_ts = time.time()
        return account

    def _saved_or_default_account_status(self, cfg: WatchConfig, *, message: str) -> BiliLoginAccount:
        is_logged_in = bool(cfg.bili_login_cookie and cfg.bilibili_cookie_source == "login")
        return BiliLoginAccount(
            is_logged_in=is_logged_in,
            uid=cfg.bili_login_uid,
            uname=cfg.bili_login_uname,
            source=cfg.bilibili_cookie_source,
            message=message,
        )

    def _format_login_runtime_status(self) -> str:
        state = self._login_runtime
        if not state.qrcode_key:
            return "idle"
        seconds_left = max(0, int(state.expires_at - time.time())) if state.expires_at > 0 else 0
        return (
            f"status={state.status} message={state.message or '-'} "
            f"expires_in={seconds_left}s url={'set' if state.url else 'empty'} "
            f"image={'set' if state.image_path else 'empty'}"
        )

    def _format_account_status(self, account: BiliLoginAccount) -> str:
        if not account.is_logged_in:
            return f"logged_in=False source={account.source or 'none'} reason={account.message or 'not_logged_in'}"
        return (
            "logged_in=True "
            f"source={account.source or 'unknown'} "
            f"uid={self._mask_uid(account.uid)} "
            f"uname={self._mask_text(account.uname)}"
        )

    def _format_channel_send_state(self, state: ChannelSendState) -> str:
        last_attempt = self._format_timestamp(state.last_attempt_ts)
        last_success = self._format_timestamp(state.last_success_ts)
        ok_text = "never" if state.ok is None else str(bool(state.ok))
        error_text = state.error or "-"
        preview = state.text_preview or "-"
        return (
            f"enabled={state.enabled} ok={ok_text} "
            f"last_attempt={last_attempt} last_success={last_success} "
            f"summary={state.summary or '-'} error={error_text} preview={preview}"
        )

    def _format_timestamp(self, raw_ts: float) -> str:
        if raw_ts <= 0:
            return "never"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(raw_ts))

    def _mask_reply_preview(self, text: str) -> str:
        normalized = self._normalize_reply(text, 30)
        return normalized or "-"

    def _mask_qrcode_key(self, value: str) -> str:
        text = str(value or "").strip()
        if len(text) <= 8:
            return "*" * len(text)
        return f"{text[:4]}...{text[-4:]}"

    def _mask_uid(self, uid: str) -> str:
        text = str(uid or "").strip()
        if len(text) <= 4:
            return text or "-"
        return f"{text[:2]}***{text[-2:]}"

    def _mask_text(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return "-"
        if len(text) <= 2:
            return text[0] + "*"
        return f"{text[0]}***{text[-1]}"

    def _sanitize_error_message(self, err: Exception | str) -> str:
        text = str(err or "").strip().replace("\r", " ").replace("\n", " ")
        if not text:
            return "unknown_error"
        text = re.sub(r"(SESSDATA|bili_jct|DedeUserID|DedeUserID__ckMd5)=([^;\\s]+)", r"\1=<hidden>", text)
        if len(text) > 160:
            text = text[:160].rstrip()
        return text

    async def _build_login_qrcode_image(self, url: str) -> str:
        return await asyncio.to_thread(self._render_login_qrcode_image, url)

    def _render_login_qrcode_image(self, url: str) -> str:
        if qrcode is None:
            logger.warning("[bili_watcher] qrcode dependency unavailable, skip QR image rendering")
            return ""
        text = str(url or "").strip()
        if not text:
            return ""
        try:
            image = qrcode.make(text)
            with tempfile.NamedTemporaryFile(
                prefix="biliwatch-login-",
                suffix=".png",
                dir="/tmp",
                delete=False,
            ) as handle:
                image.save(handle.name)
                return handle.name
        except Exception as e:
            logger.warning("[bili_watcher] build login qrcode image failed: %s", e)
            return ""

    def _cleanup_login_qrcode_image(self) -> None:
        path = str(getattr(self._login_runtime, "image_path", "") or "").strip()
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("[bili_watcher] cleanup login qrcode image failed: %s", e)

    def _resolve_effective_bili_cookie(
        self,
        *,
        manual_cookie: str,
        login_cookie: str,
    ) -> tuple[str, str]:
        if manual_cookie:
            return manual_cookie, "config"
        if login_cookie:
            return login_cookie, "login"
        return "", "none"

    def _config_get_direct(self, source: object, key: str, default: object = None) -> object:
        try:
            if isinstance(source, dict):
                return source.get(key, default)
            getter = getattr(source, "get", None)
            if callable(getter):
                return getter(key, default)
        except Exception:
            pass
        try:
            return source[key]  # type: ignore[index]
        except Exception:
            return default

    def _config_get(self, path: str, default: object = None, legacy_keys: tuple[str, ...] = ()) -> object:
        parts = [part.strip() for part in str(path or "").split(".") if part.strip()]
        current: object = self.config
        found = True
        for part in parts:
            missing = object()
            value = self._config_get_direct(current, part, missing)
            if value is missing:
                found = False
                break
            current = value
        if found:
            return current

        for legacy_key in legacy_keys:
            missing = object()
            value = self._config_get_direct(self.config, legacy_key, missing)
            if value is not missing:
                return value
        return default

    def _config_delete_path(self, path: str):
        parts = [part.strip() for part in str(path or "").split(".") if part.strip()]
        if not parts:
            return
        current = self.config
        parents: list[tuple[object, str]] = []
        for part in parts[:-1]:
            next_value = self._config_get_direct(current, part, None)
            if next_value is None:
                return
            parents.append((current, part))
            current = next_value
        last = parts[-1]
        try:
            if isinstance(current, dict):
                current.pop(last, None)
            else:
                del current[last]  # type: ignore[index]
        except Exception:
            return
        for parent, key in reversed(parents):
            child = self._config_get_direct(parent, key, None)
            if isinstance(child, dict) and not child:
                try:
                    parent.pop(key, None)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        del parent[key]  # type: ignore[index]
                    except Exception:
                        break
            else:
                break

    def _load_config(self) -> WatchConfig:
        reply_interval_seconds = self._to_int(
            self._config_get("main_loop.reply_interval_seconds", 15, legacy_keys=("reply_interval_seconds",)),
            15,
            1,
        )
        danmaku_trigger_threshold = self._to_int(
            self._config_get("main_loop.danmaku_trigger_threshold", 20, legacy_keys=("danmaku_trigger_threshold",)),
            20,
            0,
        )
        asr_trigger_threshold = self._to_int(
            self._config_get("main_loop.asr_trigger_threshold", 1, legacy_keys=("asr_trigger_threshold",)),
            1,
            0,
        )
        default_context_window_seconds = max(
            reply_interval_seconds,
            reply_interval_seconds * DEFAULT_CONTEXT_WINDOW_MULTIPLIER,
        )
        context_window_seconds = self._to_int(
            self._config_get(
                "main_loop.context_window_seconds",
                default_context_window_seconds,
                legacy_keys=("context_window_seconds",),
            ),
            default_context_window_seconds,
            1,
        )

        cookie = str(
            self._config_get("user_auth.bili_cookie", "", legacy_keys=("bili_cookie", "bilibili_cookie")) or ""
        ).strip()
        login_cookie = str(
            self._config_get("user_auth.bili_login_cookie", "", legacy_keys=("bili_login_cookie",)) or ""
        ).strip()
        login_uid = str(
            self._config_get("user_auth.bili_login_uid", "", legacy_keys=("bili_login_uid",)) or ""
        ).strip()
        login_uname = str(
            self._config_get("user_auth.bili_login_uname", "", legacy_keys=("bili_login_uname",)) or ""
        ).strip()
        login_refresh_token = str(
            self._config_get(
                "user_auth.bili_login_refresh_token",
                "",
                legacy_keys=("bili_login_refresh_token",),
            )
            or ""
        ).strip()
        login_saved_at = self._to_int(
            self._config_get("user_auth.bili_login_saved_at", 0, legacy_keys=("bili_login_saved_at",)),
            0,
            0,
        )
        cookie, cookie_source = self._resolve_effective_bili_cookie(
            manual_cookie=cookie,
            login_cookie=login_cookie,
        )

        return WatchConfig(
            enabled=self._to_bool(self._config_get("global.enabled", True, legacy_keys=("enabled",)), True),
            debug=self._to_bool(self._config_get("global.debug", False, legacy_keys=("debug",)), False),
            room_id=self._to_int(self._config_get("global.room_id", 0, legacy_keys=("room_id",)), 0, 0),
            reply_interval_seconds=reply_interval_seconds,
            context_window_seconds=context_window_seconds,
            danmaku_trigger_threshold=danmaku_trigger_threshold,
            asr_trigger_threshold=asr_trigger_threshold,
            target_umo=str(self._config_get("sender.target_umo", "", legacy_keys=("target_umo",)) or "").strip(),
            target_platform_id=str(
                self._config_get("sender.target_platform_id", "default", legacy_keys=("target_platform_id",))
                or "default"
            ).strip(),
            target_type=str(
                self._config_get("sender.target_type", "group", legacy_keys=("target_type",)) or "group"
            ).strip(),
            target_id=str(self._config_get("sender.target_id", "", legacy_keys=("target_id",)) or "").strip(),
            generation_provider_id=str(
                self._config_get("generation.provider_id", "", legacy_keys=("generation_provider_id", "provider_id"))
                or ""
            ).strip(),
            generation_persona_id=str(
                self._config_get("generation.persona_id", "", legacy_keys=("generation_persona_id", "persona_id"))
                or ""
            ).strip(),
            generation_prompt_template=str(
                self._config_get(
                    "generation.prompt_template",
                    DEFAULT_FUSED_PROMPT_TEMPLATE,
                    legacy_keys=("generation_prompt_template", "prompt_template"),
                )
                or DEFAULT_FUSED_PROMPT_TEMPLATE
            ),
            max_reply_chars=self._to_int(
                self._config_get("generation.max_reply_chars", 60, legacy_keys=("max_reply_chars",)),
                60,
                10,
            ),
            bilibili_cookie=cookie,
            bilibili_cookie_source=cookie_source,
            sync_to_bilibili_live=self._to_bool(
                self._config_get("global.sync_to_bilibili_live", False, legacy_keys=("sync_to_bilibili_live",)),
                False,
            ),
            bili_login_cookie=login_cookie,
            bili_login_uid=login_uid,
            bili_login_uname=login_uname,
            bili_login_refresh_token=login_refresh_token,
            bili_login_saved_at=login_saved_at,
            pipeline_mode=self._normalize_pipeline_mode(
                self._config_get("global.pipeline_mode", 2, legacy_keys=("pipeline_mode",))
            ),
            use_realtime_danmaku_ws=INTERNAL_USE_REALTIME_DANMAKU_WS,
            danmu_ws_auth_mode=INTERNAL_DANMU_WS_AUTH_MODE,
            allow_buvid3_only=INTERNAL_ALLOW_BUVID3_ONLY,
            wbi_sign_enabled=INTERNAL_WBI_SIGN_ENABLED,
            audio_pull_protocol=str(
                self._config_get("asr.audio_pull_protocol", "http_flv", legacy_keys=("audio_pull_protocol",))
                or "http_flv"
            ).strip(),
            audio_pull_api_preference=INTERNAL_AUDIO_PULL_API_PREFERENCE,
            audio_http_headers_enabled=INTERNAL_AUDIO_HTTP_HEADERS_ENABLED,
            ffmpeg_path=str(self._config_get("asr.ffmpeg_path", "ffmpeg", legacy_keys=("ffmpeg_path",)) or "ffmpeg").strip(),
            audio_sample_rate=self._to_int(
                self._config_get("asr.audio_sample_rate", 16000, legacy_keys=("audio_sample_rate",)),
                16000,
                8000,
            ),
            asr_backend=str(
                self._config_get("asr.asr_backend", "sherpa_onnx_rknn", legacy_keys=("asr_backend",))
                or "sherpa_onnx_rknn"
            ).strip(),
            asr_model_dir=self._resolve_plugin_path(
                str(
                    self._config_get("asr.asr_model_dir", DEFAULT_ASR_MODEL_DIR, legacy_keys=("asr_model_dir",))
                    or DEFAULT_ASR_MODEL_DIR
                ).strip(),
                DEFAULT_ASR_MODEL_DIR,
            ),
            asr_vad_model_path=self._resolve_plugin_path(
                str(
                    self._config_get(
                        "asr.asr_vad_model_path",
                        DEFAULT_ASR_VAD_MODEL_PATH,
                        legacy_keys=("asr_vad_model_path",),
                    )
                    or DEFAULT_ASR_VAD_MODEL_PATH
                ).strip(),
                DEFAULT_ASR_VAD_MODEL_PATH,
            ),
            asr_vad_threshold=self._to_float(
                self._config_get("asr.asr_vad_threshold", 0.3, legacy_keys=("asr_vad_threshold",)),
                0.3,
            ),
            asr_vad_min_silence_duration=self._to_float(
                self._config_get(
                    "asr.asr_vad_min_silence_duration",
                    0.35,
                    legacy_keys=("asr_vad_min_silence_duration",),
                ),
                0.35,
            ),
            asr_vad_min_speech_duration=self._to_float(
                self._config_get(
                    "asr.asr_vad_min_speech_duration",
                    0.25,
                    legacy_keys=("asr_vad_min_speech_duration",),
                ),
                0.25,
            ),
            asr_vad_max_speech_duration=self._to_float(
                self._config_get(
                    "asr.asr_vad_max_speech_duration",
                    20.0,
                    legacy_keys=("asr_vad_max_speech_duration",),
                ),
                20.0,
            ),
            asr_sense_voice_language=str(
                self._config_get(
                    "asr.asr_sense_voice_language",
                    "auto",
                    legacy_keys=("asr_sense_voice_language",),
                )
                or "auto"
            ).strip(),
            asr_sense_voice_use_itn=INTERNAL_ASR_SENSE_VOICE_USE_ITN,
            asr_runtime_probe_required=INTERNAL_ASR_RUNTIME_PROBE_REQUIRED,
            asr_threads=self._to_int(
                self._config_get("asr.asr_threads", 1, legacy_keys=("asr_threads",)),
                1,
                -4,
            ),
            singer_mode_enabled=self._to_bool(
                self._config_get(
                    "singer.singer_mode_enabled",
                    self._config_get("other.singer_mode_enabled", True, legacy_keys=("singer_mode_enabled",)),
                ),
                True,
            ),
            singer_mode_keywords=self._to_string_list(
                self._config_get(
                    "singer.singer_mode_keywords",
                    self._config_get(
                        "other.singer_mode_keywords",
                        list(DEFAULT_SINGER_KEYWORDS),
                        legacy_keys=("singer_mode_keywords",),
                    ),
                ),
                list(DEFAULT_SINGER_KEYWORDS),
            ),
            singer_mode_window_seconds=self._to_int(
                self._config_get(
                    "singer.singer_mode_window_seconds",
                    self._config_get(
                        "other.singer_mode_window_seconds",
                        20,
                        legacy_keys=("singer_mode_window_seconds",),
                    ),
                ),
                20,
                0,
            ),
            singer_mode_instruction=str(
                self._config_get(
                    "singer.singer_mode_instruction",
                    self._config_get(
                        "other.singer_mode_instruction",
                        DEFAULT_SINGER_MODE_INSTRUCTION,
                        legacy_keys=("singer_mode_instruction",),
                    ),
                )
                or ""
            ).strip(),
        )

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

    def _to_string_list(self, value: object, default: list[str]) -> list[str]:
        raw_items: list[object]
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, tuple):
            raw_items = list(value)
        elif isinstance(value, str):
            raw_items = [part.strip() for part in value.split(",")]
        else:
            raw_items = list(default)

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned or list(default)

    def _log_warn_throttled(self, msg: str, interval_seconds: int = 60):
        now = time.time()
        if now - self._last_warn_ts >= interval_seconds:
            logger.warning(msg)
            self._last_warn_ts = now

    def _set_integer_config_from_command(
        self,
        event: AstrMessageEvent,
        *,
        explicit_value: str,
        config_key: str,
        label: str,
        min_value: int,
        example: str,
    ) -> str:
        raw = str(explicit_value or "").strip()
        if not raw:
            parts = str(getattr(event, "message_str", "") or "").strip().split()
            if len(parts) >= 3:
                raw = parts[-1].strip()

        try:
            parsed = int(raw)
            if parsed < min_value:
                raise ValueError
        except Exception:
            min_text = "0 或更大的整数" if min_value == 0 else f"{min_value} 或更大的整数"
            return f"{label} 无效，请输入{min_text}。例如：{example}"

        self._set_config_value(config_key, parsed)
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        return f"已将 {label} 设置为 {parsed} {suffix}"

    def _set_config_value(self, key: str, value: object):
        parts = [part.strip() for part in str(key or "").split(".") if part.strip()]
        if not parts:
            return
        try:
            current = self.config
            for part in parts[:-1]:
                next_value = self._config_get_direct(current, part, None)
                if not isinstance(next_value, dict):
                    next_value = {}
                    if isinstance(current, dict):
                        current[part] = next_value
                    else:
                        current[part] = next_value  # type: ignore[index]
                current = next_value
            last = parts[-1]
            if isinstance(current, dict):
                current[last] = value
            else:
                current[last] = value  # type: ignore[index]
        except Exception:
            pass

    def _drop_hidden_legacy_config_keys(self):
        for key in HIDDEN_LEGACY_CONFIG_KEYS:
            try:
                if hasattr(self.config, "pop"):
                    self.config.pop(key, None)
                    continue
            except Exception:
                pass
            try:
                del self.config[key]
            except Exception:
                pass

    def _save_config_if_possible(self) -> bool:
        self._drop_hidden_legacy_config_keys()
        if hasattr(self.config, "save_config"):
            try:
                self.config.save_config()
                return True
            except Exception as e:
                logger.warning(f"[bili_watcher] save_config failed: {e}")
        return False

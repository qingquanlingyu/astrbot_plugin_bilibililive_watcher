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

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
DEFAULT_COOKIE_FILE = "~/.bilibili-cookie.json"


@dataclass(slots=True)
class DanmakuItem:
    uid: str
    nickname: str
    text: str
    ts: float
    timeline: str
    dedup_key: str


@dataclass(slots=True)
class WatchConfig:
    enabled: bool
    room_id: int
    window_seconds: int
    poll_interval_seconds: int
    trigger_threshold: int
    trigger_cooldown_seconds: int
    target_umo: str
    target_platform_id: str
    target_type: str
    target_id: str
    max_context_danmaku: int
    max_history_messages: int
    max_reply_chars: int
    inject_bili_send_hint: bool
    bilibili_cookie: str
    bilibili_cookie_file: str
    auto_load_cookie_from_file: bool


@register(
    "astrbot_plugin_bilibililive_watcher",
    "YourName",
    "监听B站直播弹幕热度并触发模型生成短弹幕发送到指定会话。",
    "1.0.0",
)
class BilibiliLiveWatcherPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.context = context
        self.config = config or {}
        self._task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._buffer: list[DanmakuItem] = []
        self._seen: dict[str, float] = {}
        self._last_trigger_ts = 0.0
        self._last_warn_ts = 0.0
        self._real_room_id: int | None = None
        self._real_room_id_source: int | None = None

    async def initialize(self):
        if self._task and not self._task.done():
            return
        self._session = aiohttp.ClientSession(headers={"User-Agent": DEFAULT_UA})
        self._task = asyncio.create_task(self._watch_loop(), name="bili-live-watcher")
        logger.info("[bili_watcher] initialized")

    async def terminate(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
        logger.info("[bili_watcher] terminated")

    @filter.command("biliwatch_status")
    async def biliwatch_status(self, event: AstrMessageEvent):
        """查看当前 B 站弹幕监听配置状态"""
        cfg = self._load_config()
        target_umo = self._resolve_target_umo(cfg)
        platform_ids = self._get_available_platform_ids()
        last_trigger = "never"
        if self._last_trigger_ts > 0:
            last_trigger = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self._last_trigger_ts)
            )
        yield event.plain_result(
            "\n".join(
                [
                    f"enabled: {cfg.enabled}",
                    f"room_id: {cfg.room_id}",
                    f"target: {target_umo or '(未配置)'}",
                    f"platform_ids: {platform_ids or '[]'}",
                    f"window_seconds: {cfg.window_seconds}",
                    f"poll_interval_seconds: {cfg.poll_interval_seconds}",
                    f"trigger_threshold: {cfg.trigger_threshold}",
                    f"buffer_size: {len(self._buffer)}",
                    f"last_trigger: {last_trigger}",
                ]
            )
        )

    @filter.command("biliwatch_set_room", alias={"biliwatch room", "设置弹幕监听直播间"})
    async def biliwatch_set_room(self, event: AstrMessageEvent, room_id: str = ""):
        """设置监听的 B 站直播间号。用法：/biliwatch_set_room 22642754"""
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
            yield event.plain_result("room_id 无效，请输入正整数。例如：/biliwatch_set_room 22642754")
            return

        self._set_config_value("room_id", room_id_int)
        self._real_room_id = None
        self._real_room_id_source = None
        saved = self._save_config_if_possible()
        suffix = "（已保存）" if saved else "（运行时已生效，未持久化）"
        yield event.plain_result(f"已将监听直播间设置为 {room_id_int} {suffix}")

    @filter.command("biliwatch_bind_here", alias={"biliwatch bind", "绑定弹幕发送到当前会话"})
    async def biliwatch_bind_here(self, event: AstrMessageEvent):
        """将消息发送目标绑定为当前指令所在会话。"""
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

    async def _watch_loop(self):
        while True:
            cfg = self._load_config()
            try:
                await self._tick(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[bili_watcher] tick failed: {e}")
            await asyncio.sleep(max(3, cfg.poll_interval_seconds))

    async def _tick(self, cfg: WatchConfig):
        if not cfg.enabled:
            return
        if cfg.room_id <= 0:
            self._log_warn_throttled("[bili_watcher] room_id 未配置，已跳过轮询")
            return

        target_umo = self._resolve_target_umo(cfg)
        if not target_umo:
            self._log_warn_throttled("[bili_watcher] 发送目标未配置，已跳过轮询")
            return

        room_id = await self._resolve_real_room_id(cfg.room_id, cfg.bilibili_cookie)
        new_msgs = await self._fetch_new_danmaku(room_id, cfg.bilibili_cookie)
        if not new_msgs:
            self._prune_old(cfg.window_seconds)
            return

        self._buffer.extend(new_msgs)
        self._prune_old(cfg.window_seconds)
        if len(self._buffer) < cfg.trigger_threshold:
            return

        now = time.time()
        if now - self._last_trigger_ts < cfg.trigger_cooldown_seconds:
            return

        recent = self._buffer[-cfg.max_context_danmaku :]
        reply = await self._generate_short_reply(
            cfg=cfg,
            target_umo=target_umo,
            room_id=room_id,
            danmaku_items=recent,
        )
        self._last_trigger_ts = now
        self._buffer.clear()

        if not reply:
            logger.warning("[bili_watcher] 模型未生成有效内容，本次不发送")
            return

        ok = await self.context.send_message(
            target_umo,
            MessageChain().message(reply),
        )
        if ok:
            logger.info(
                f"[bili_watcher] sent to {target_umo}, room={room_id}, text={reply[:120]}"
            )
        else:
            logger.warning(f"[bili_watcher] send_message failed, target={target_umo}")

    async def _resolve_real_room_id(self, room_id: int, cookie: str) -> int:
        if self._real_room_id_source == room_id and self._real_room_id:
            return self._real_room_id

        session = self._session
        if not session:
            return room_id

        try:
            headers = self._make_http_headers(cookie=cookie, room_id=room_id)
            async with session.get(
                "https://api.live.bilibili.com/room/v1/Room/room_init",
                params={"id": room_id},
                headers=headers,
                timeout=10,
            ) as resp:
                data = await resp.json(content_type=None)
            if data.get("code") == 0:
                actual_id = int(data.get("data", {}).get("room_id", room_id))
                self._real_room_id = actual_id
                self._real_room_id_source = room_id
                return actual_id
        except Exception as e:
            logger.warning(f"[bili_watcher] resolve real room id failed: {e}")

        self._real_room_id = room_id
        self._real_room_id_source = room_id
        return room_id

    async def _fetch_new_danmaku(self, room_id: int, cookie: str) -> list[DanmakuItem]:
        session = self._session
        if not session:
            return []

        headers = self._make_http_headers(cookie=cookie, room_id=room_id)
        async with session.get(
            "https://api.live.bilibili.com/xlive/web-room/v1/dM/gethistory",
            params={"roomid": room_id},
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if payload.get("code") != 0:
            raise RuntimeError(
                f"gethistory failed: code={payload.get('code')} msg={payload.get('message')}"
            )

        raw_data = payload.get("data", {})
        raw_items = []
        for key in ("room", "admin"):
            value = raw_data.get(key, [])
            if isinstance(value, list):
                raw_items.extend(value)

        now = time.time()
        items: list[DanmakuItem] = []
        for row in raw_items:
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            uid = str(row.get("uid", ""))
            nickname = str(row.get("nickname", "") or uid)
            timeline = str(row.get("timeline", "")).strip()
            dedup_key = f"{uid}|{timeline}|{text}"
            if dedup_key in self._seen:
                continue

            self._seen[dedup_key] = now
            items.append(
                DanmakuItem(
                    uid=uid,
                    nickname=nickname,
                    text=text,
                    ts=now,
                    timeline=timeline,
                    dedup_key=dedup_key,
                )
            )

        items.sort(key=lambda x: x.ts)
        return items

    async def _generate_short_reply(
        self,
        cfg: WatchConfig,
        target_umo: str,
        room_id: int,
        danmaku_items: list[DanmakuItem],
    ) -> str:
        try:
            provider = self.context.get_using_provider(umo=target_umo)
        except TypeError:
            provider = self.context.get_using_provider()
        if not provider:
            provider = self.context.get_using_provider()
        if not provider:
            logger.warning("[bili_watcher] no provider available")
            return ""

        conversation, contexts = await self._get_recent_contexts(
            target_umo=target_umo,
            max_messages=cfg.max_history_messages,
        )
        system_prompt = await self._get_system_prompt(target_umo, conversation)
        prompt = self._build_prompt(cfg, room_id, danmaku_items)

        llm_resp = None
        try:
            llm_resp = await provider.text_chat(
                prompt=prompt,
                contexts=contexts,
                system_prompt=system_prompt or "",
            )
        except TypeError:
            try:
                llm_resp = await provider.text_chat(
                    prompt=prompt,
                    system_prompt=system_prompt or "",
                )
            except TypeError:
                llm_resp = await provider.text_chat(prompt)
        except Exception as e:
            logger.warning(f"[bili_watcher] provider.text_chat failed: {e}")
            if self._is_tool_context_mismatch_error(e):
                logger.warning(
                    "[bili_watcher] 检测到 tool 上下文不匹配，自动降级为无 contexts 重试"
                )
                llm_resp = await self._call_provider_without_contexts(
                    provider=provider,
                    prompt=prompt,
                    system_prompt=system_prompt or "",
                )

        text = self._extract_llm_text(llm_resp)
        if not text:
            return ""
        return self._normalize_reply(text, cfg.max_reply_chars)

    async def _get_recent_contexts(
        self, target_umo: str, max_messages: int
    ) -> tuple[object | None, list[dict]]:
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
                # 丢弃 tool / function 等消息，避免 tool_call_id 链路断裂导致 400
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

    async def _call_provider_without_contexts(
        self, provider: object, prompt: str, system_prompt: str
    ) -> object | None:
        try:
            return await provider.text_chat(
                prompt=prompt,
                system_prompt=system_prompt,
            )
        except TypeError:
            return await provider.text_chat(prompt)
        except Exception as e:
            logger.warning(f"[bili_watcher] retry without contexts failed: {e}")
            return None

    def _is_tool_context_mismatch_error(self, err: Exception) -> bool:
        text = str(err).lower()
        keywords = (
            "tool id() not found",
            "tool result's tool id",
            "tool_call_id",
            "tool result",
        )
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

    def _build_prompt(
        self, cfg: WatchConfig, room_id: int, danmaku_items: list[DanmakuItem]
    ) -> str:
        lines = []
        for item in danmaku_items:
            who = item.nickname or item.uid or "观众"
            lines.append(f"{who}: {item.text}")
        danmaku_text = "\n".join(lines)

        hint_line = ""
        if cfg.inject_bili_send_hint:
            hint_line = (
                "4) 若你具备 `bili_danmu_sender` skill，请使用该skill直接"
                "把弹幕发送到该直播间；若不具备则忽略，不要在输出中提及工具。\n"
            )

        return (
            f"你需要基于最近 {cfg.window_seconds} 秒内的B站直播间弹幕热度，"
            "生成1条可以发到群聊/私聊的简短互动弹幕。\n"
            f"直播间号: {room_id}\n"
            f"弹幕条数: {len(danmaku_items)}\n\n"
            "输出要求:\n"
            "1) 只输出1句话，单行，不要解释，不要Markdown。\n"
            "2) 语气贴合当前人设，简短自然，10~30字。\n"
            "3) 不要复读原弹幕，避免攻击性内容。\n"
            f"{hint_line}\n"
            "最近弹幕样本:\n"
            f"{danmaku_text}\n"
        )

    def _extract_llm_text(self, resp: object) -> str:
        if resp is None:
            return ""
        for key in ("completion_text", "completion", "text", "content"):
            value = getattr(resp, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _normalize_reply(self, text: str, max_chars: int) -> str:
        text = text.strip().replace("\r", " ").replace("\n", " ")
        text = text.strip("`").strip()
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1].strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip()
        return text

    def _prune_old(self, window_seconds: int):
        now = time.time()
        cutoff = now - max(1, window_seconds)
        self._buffer = [item for item in self._buffer if item.ts >= cutoff]
        seen_ttl = max(600, window_seconds * 6)
        seen_cutoff = now - seen_ttl
        self._seen = {k: ts for k, ts in self._seen.items() if ts >= seen_cutoff}

    def _make_http_headers(self, cookie: str, room_id: int) -> dict[str, str]:
        headers = {
            "Origin": "https://live.bilibili.com",
            "Referer": f"https://live.bilibili.com/{room_id}",
            "User-Agent": DEFAULT_UA,
        }
        if cookie:
            headers["Cookie"] = cookie
        return headers

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

            # 兼容只填了群号/私聊号的情况
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
        # 对 이미是标准格式的场景做兼容
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

    def _load_config(self) -> WatchConfig:
        window_seconds = self._to_int(self.config.get("window_seconds", 0), 0, 1)
        if window_seconds <= 0:
            window_minutes = self._to_float(self.config.get("window_minutes", 1), 1.0)
            window_seconds = max(10, int(window_minutes * 60))

        poll_interval_seconds = self._to_int(
            self.config.get("poll_interval_seconds", window_seconds),
            window_seconds,
            3,
        )
        threshold = self._to_int(self.config.get("trigger_threshold", 20), 20, 1)
        cooldown = self._to_int(
            self.config.get("trigger_cooldown_seconds", window_seconds),
            window_seconds,
            1,
        )

        cookie = str(self.config.get("bilibili_cookie", "") or "").strip()
        cookie_file = str(
            self.config.get("bilibili_cookie_file", DEFAULT_COOKIE_FILE)
            or DEFAULT_COOKIE_FILE
        ).strip()
        auto_load_cookie = self._to_bool(
            self.config.get("auto_load_cookie_from_file", True), True
        )
        if not cookie and auto_load_cookie:
            cookie = self._load_cookie_from_file(cookie_file)

        return WatchConfig(
            enabled=self._to_bool(self.config.get("enabled", True), True),
            room_id=self._to_int(self.config.get("room_id", 0), 0, 0),
            window_seconds=window_seconds,
            poll_interval_seconds=poll_interval_seconds,
            trigger_threshold=threshold,
            trigger_cooldown_seconds=cooldown,
            target_umo=str(self.config.get("target_umo", "") or "").strip(),
            target_platform_id=str(
                self.config.get("target_platform_id", "default") or "default"
            ).strip(),
            target_type=str(self.config.get("target_type", "group") or "group").strip(),
            target_id=str(self.config.get("target_id", "") or "").strip(),
            max_context_danmaku=self._to_int(
                self.config.get("max_context_danmaku", 40), 40, 5
            ),
            max_history_messages=self._to_int(
                self.config.get("max_history_messages", 12), 12, 0
            ),
            max_reply_chars=self._to_int(self.config.get("max_reply_chars", 60), 60, 10),
            inject_bili_send_hint=self._to_bool(
                self.config.get("inject_bili_send_hint", True), True
            ),
            bilibili_cookie=cookie,
            bilibili_cookie_file=cookie_file,
            auto_load_cookie_from_file=auto_load_cookie,
        )

    def _load_cookie_from_file(self, cookie_file: str) -> str:
        path = Path(cookie_file).expanduser()
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cookie = str(data.get("cookie", "") or "").strip()
            return cookie
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

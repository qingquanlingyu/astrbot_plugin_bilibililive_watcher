from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import zlib
from typing import Awaitable, Callable

from aiohttp import WSMsgType
try:
    from astrbot.api import logger
except Exception:  # pragma: no cover
    logger = logging.getLogger("bili_watcher")

try:  # pragma: no cover
    from .bili_http import BiliApiError, BiliHttpClient, DanmuInfoAuthError
    from .models import DanmakuItem
except ImportError:  # pragma: no cover
    from bili_http import BiliApiError, BiliHttpClient, DanmuInfoAuthError
    from models import DanmakuItem

try:
    import brotli
except Exception:  # pragma: no cover
    brotli = None


DanmakuCallback = Callable[[DanmakuItem], Awaitable[None]]


class DanmakuRealtimeClient:
    def __init__(
        self,
        *,
        http_client: BiliHttpClient,
        room_id: int,
        cookie: str,
        wbi_cookie: str,
        ws_require_wbi_sign: bool,
        on_danmaku: DanmakuCallback,
        heartbeat_interval: int = 30,
    ):
        self._http = http_client
        self._room_id = room_id
        self._cookie = cookie
        self._wbi_cookie = wbi_cookie
        self._ws_require_wbi_sign = ws_require_wbi_sign
        self._on_danmaku = on_danmaku
        self._heartbeat_interval = max(10, heartbeat_interval)
        self._task: asyncio.Task | None = None
        self._protover = 3 if brotli is not None else 2
        self.connected = False
        self.fatal_error: str = ""

    async def start(self):
        if self._task and not self._task.done():
            return
        self.fatal_error = ""
        self._task = asyncio.create_task(self._run(), name="bili-danmaku-ws")

    async def stop(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        self.connected = False

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done())

    async def _run(self):
        backoff = 1
        while True:
            try:
                await self._connect_once()
                backoff = 1
            except asyncio.CancelledError:
                raise
            except DanmuInfoAuthError as e:
                self.connected = False
                self.fatal_error = str(e)
                logger.warning(f"[bili_watcher] ws auth blocked: {e}")
                return
            except Exception as e:
                self.connected = False
                logger.warning(f"[bili_watcher] ws disconnected: {e}")
            await asyncio.sleep(backoff)
            backoff = min(30, backoff * 2)

    async def _connect_once(self):
        data = await self._http.get_danmu_info(
            room_id=self._room_id,
            cookie=self._cookie,
            wbi_cookie=self._wbi_cookie,
            ws_require_wbi_sign=self._ws_require_wbi_sign,
        )
        token = str(data.get("token", "") or "")
        hosts = data.get("host_list", []) or []
        if not token or not hosts:
            raise BiliApiError("getDanmuInfo 成功但缺少 token/host_list")

        host = hosts[0]
        ws_host = str(host.get("host", "") or "broadcastlv.chat.bilibili.com")
        wss_port = int(host.get("wss_port", 443) or 443)
        url = f"wss://{ws_host}:{wss_port}/sub"

        ws = await self._http.session.ws_connect(
            url,
            heartbeat=None,
            receive_timeout=70,
            timeout=10,
        )
        hb_task = asyncio.create_task(self._heartbeat_loop(ws), name="bili-ws-heartbeat")
        try:
            await ws.send_bytes(self._pack_auth(self._room_id, token))
            self.connected = True
            logger.info(f"[bili_watcher] ws connected: room={self._room_id} host={ws_host}")
            async for msg in ws:
                if msg.type == WSMsgType.CLOSED:
                    break
                if msg.type == WSMsgType.ERROR:
                    break
                if msg.type == WSMsgType.BINARY:
                    await self._handle_binary(msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._handle_text(msg.data)
        finally:
            self.connected = False
            hb_task.cancel()
            await asyncio.gather(hb_task, return_exceptions=True)
            await ws.close()

    async def _heartbeat_loop(self, ws):
        packet = self._pack_packet(b"[object Object]", op=2, ver=1)
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            await ws.send_bytes(packet)

    async def _handle_text(self, raw: str):
        raw = str(raw or "").strip()
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            return
        await self._dispatch_event(payload)

    async def _handle_binary(self, data: bytes):
        for op, ver, body in self._iter_packets(data):
            if op != 5:
                continue
            if ver == 0 or ver == 1:
                try:
                    payload = json.loads(body.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                await self._dispatch_event(payload)

    async def _dispatch_event(self, payload: dict):
        cmd = str(payload.get("cmd", "") or "")
        cmd_main = cmd.split(":", 1)[0]
        now = time.time()

        if cmd_main == "DANMU_MSG":
            info = payload.get("info", []) or []
            text = ""
            uid = ""
            nickname = ""
            timeline = ""
            dedup_base = ""
            if len(info) > 1:
                text = str(info[1] or "").strip()
            if len(info) > 2 and isinstance(info[2], list):
                user = info[2]
                if len(user) > 0:
                    uid = str(user[0] or "")
                if len(user) > 1:
                    nickname = str(user[1] or "")
            if len(info) > 0 and isinstance(info[0], list):
                head = info[0]
                if len(head) > 4:
                    dedup_base = str(head[4] or "")
                if len(head) > 5:
                    timeline = str(head[5] or "")
            if not text:
                return
            dedup_key = dedup_base or f"ws|danmu|{uid}|{timeline}|{text}"
            await self._on_danmaku(
                DanmakuItem(
                    uid=uid,
                    nickname=nickname or uid or "观众",
                    text=text,
                    ts=now,
                    timeline=timeline,
                    dedup_key=dedup_key,
                    event_type="danmu",
                    source="ws",
                )
            )
            return

        return

    def _iter_packets(self, data: bytes):
        offset = 0
        total = len(data)
        while offset + 16 <= total:
            pack_len, header_len, ver, op, _seq = struct.unpack(
                ">IHHII", data[offset : offset + 16]
            )
            if pack_len < 16:
                break
            body_start = offset + header_len
            body_end = offset + pack_len
            if body_end > total:
                break
            body = data[body_start:body_end]
            offset += pack_len

            if ver == 2:
                try:
                    decompressed = zlib.decompress(body)
                except Exception:
                    continue
                yield from self._iter_packets(decompressed)
                continue
            if ver == 3:
                if brotli is None:
                    continue
                try:
                    decompressed = brotli.decompress(body)
                except Exception:
                    continue
                yield from self._iter_packets(decompressed)
                continue
            yield op, ver, body

    def _pack_auth(self, room_id: int, token: str) -> bytes:
        body = json.dumps(
            {
                "uid": 0,
                "roomid": room_id,
                "protover": self._protover,
                "platform": "web",
                "type": 2,
                "key": token,
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return self._pack_packet(body, op=7, ver=1)

    def _pack_packet(self, body: bytes, op: int, ver: int) -> bytes:
        header_len = 16
        pack_len = header_len + len(body)
        return struct.pack(">IHHII", pack_len, header_len, ver, op, 1) + body

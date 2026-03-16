from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import aiohttp

try:  # pragma: no cover
    from .bili_auth import BiliWbiSigner
    from .models import DanmakuItem
except ImportError:  # pragma: no cover
    from bili_auth import BiliWbiSigner
    from models import DanmakuItem

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class BiliApiError(RuntimeError):
    pass


class DanmuInfoAuthError(BiliApiError):
    pass


@dataclass(slots=True)
class RoomPromptMeta:
    room_title: str = ""
    anchor_name: str = ""
    live_status: int | None = None


class BiliHttpClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        wbi_signer: BiliWbiSigner | None = None,
    ):
        self._session = session
        self._wbi_signer = wbi_signer or BiliWbiSigner(session)

    @property
    def session(self) -> aiohttp.ClientSession:
        return self._session

    async def resolve_real_room_id(self, room_id: int, cookie: str = "") -> int:
        headers = self.make_headers(cookie=cookie, room_id=room_id)
        async with self._session.get(
            "https://api.live.bilibili.com/room/v1/Room/room_init",
            params={"id": room_id},
            headers=headers,
            timeout=10,
        ) as resp:
            data = await resp.json(content_type=None)
        if data.get("code") != 0:
            return room_id
        return int(data.get("data", {}).get("room_id", room_id) or room_id)

    async def get_room_prompt_meta(self, room_id: int, cookie: str = "") -> RoomPromptMeta:
        headers = self.make_headers(cookie=cookie, room_id=room_id)
        room_title = ""
        anchor_name = ""
        uid = 0
        live_status: int | None = None

        async with self._session.get(
            "https://api.live.bilibili.com/room/v1/Room/get_info",
            params={"room_id": room_id},
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)
        if payload.get("code") == 0:
            data = payload.get("data", {}) or {}
            room_title = str(data.get("title", "") or "").strip()
            uid = int(data.get("uid", 0) or 0)
            raw_live_status = data.get("live_status")
            if raw_live_status is not None:
                try:
                    live_status = int(raw_live_status)
                except (TypeError, ValueError):
                    live_status = None

        async with self._session.get(
            "https://api.live.bilibili.com/live_user/v1/UserInfo/get_anchor_in_room",
            params={"roomid": room_id},
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)
        if payload.get("code") == 0:
            data = payload.get("data", {}) or {}
            info = data.get("info", {}) or {}
            anchor_name = str(info.get("uname", "") or "").strip()

        if not anchor_name and uid > 0:
            async with self._session.get(
                "https://api.live.bilibili.com/live_user/v1/Master/info",
                params={"uid": uid},
                headers=headers,
                timeout=10,
            ) as resp:
                payload = await resp.json(content_type=None)
            if payload.get("code") == 0:
                data = payload.get("data", {}) or {}
                info = data.get("info", {}) or {}
                anchor_name = str(info.get("uname", "") or "").strip()

        return RoomPromptMeta(
            room_title=room_title,
            anchor_name=anchor_name,
            live_status=live_status,
        )

    async def get_history_danmaku(self, room_id: int, cookie: str = "") -> list[DanmakuItem]:
        headers = self.make_headers(cookie=cookie, room_id=room_id)
        async with self._session.get(
            "https://api.live.bilibili.com/xlive/web-room/v1/dM/gethistory",
            params={"roomid": room_id},
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if payload.get("code") != 0:
            raise BiliApiError(
                f"gethistory failed: code={payload.get('code')} msg={payload.get('message')}"
            )

        raw_data = payload.get("data", {})
        raw_items: list[dict[str, Any]] = []
        for key in ("room", "admin"):
            value = raw_data.get(key, [])
            if isinstance(value, list):
                raw_items.extend(value)

        now = time.time()
        items: list[DanmakuItem] = []
        for row in raw_items:
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            uid = str(row.get("uid", "") or "")
            nickname = str(row.get("nickname", "") or uid)
            timeline = str(row.get("timeline", "") or "").strip()
            dedup_key = f"{uid}|{timeline}|{text}"
            items.append(
                DanmakuItem(
                    uid=uid,
                    nickname=nickname,
                    text=text,
                    ts=now,
                    timeline=timeline,
                    dedup_key=dedup_key,
                    event_type="danmu",
                    source="history",
                )
            )
        items.sort(key=lambda x: x.ts)
        return items

    async def get_danmu_info(
        self,
        room_id: int,
        cookie: str = "",
        wbi_cookie: str = "",
        ws_require_wbi_sign: bool = True,
    ) -> dict[str, Any]:
        use_cookie = wbi_cookie or cookie
        headers = self.make_headers(cookie=use_cookie, room_id=room_id)
        params: dict[str, Any] = {"id": room_id, "type": 0}
        if ws_require_wbi_sign:
            params = await self._wbi_signer.sign(params)
        async with self._session.get(
            "https://api.live.bilibili.com/xlive/web-room/v1/index/getDanmuInfo",
            params=params,
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        code = int(payload.get("code", -1))
        if code == 0:
            return payload.get("data", {}) or {}
        if code == -352:
            raise DanmuInfoAuthError(
                "getDanmuInfo 返回 -352（可能需要 WBI 签名/风控校验）"
            )
        if ws_require_wbi_sign:
            raise BiliApiError(
                f"getDanmuInfo failed: code={code} msg={payload.get('message')}"
            )
        return {}

    async def get_room_play_urls(
        self,
        room_id: int,
        cookie: str = "",
        pull_protocol: str = "http_flv",
        api_preference: str = "getRoomPlayInfo",
    ) -> list[str]:
        headers = self.make_headers(cookie=cookie, room_id=room_id)
        urls: list[str] = []

        prefer_legacy = str(api_preference or "").strip() == "playUrl"
        if prefer_legacy:
            urls.extend(await self._get_legacy_play_url_candidates(room_id=room_id, headers=headers))
            if urls:
                return self._dedupe_urls(urls)

        payload = await self._get_room_play_info_payload(room_id=room_id, headers=headers)
        data = payload.get("data", {}) or {}
        urls.extend(self._extract_play_info_urls(data, pull_protocol=pull_protocol))
        if urls:
            return self._dedupe_urls(urls)

        urls.extend(await self._get_legacy_play_url_candidates(room_id=room_id, headers=headers))
        return self._dedupe_urls(urls)

    async def _get_room_play_info_payload(
        self, room_id: int, headers: dict[str, str]
    ) -> dict[str, Any]:
        params = {
            "room_id": room_id,
            "protocol": "0,1",
            "format": "0,1,2",
            "codec": "0,1",
            "qn": 150,
            "platform": "web",
            "ptype": 8,
        }
        async with self._session.get(
            "https://api.live.bilibili.com/xlive/web-room/v2/index/getRoomPlayInfo",
            params=params,
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if payload.get("code") != 0:
            raise BiliApiError(
                f"getRoomPlayInfo failed: code={payload.get('code')} msg={payload.get('message')}"
            )
        return payload

    async def _get_legacy_play_url_candidates(
        self, room_id: int, headers: dict[str, str]
    ) -> list[str]:
        async with self._session.get(
            "https://api.live.bilibili.com/room/v1/Room/playUrl",
            params={"cid": room_id, "platform": "web", "qn": 10000},
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if payload.get("code") != 0:
            return []

        data = payload.get("data", {}) or {}
        urls: list[str] = []
        for row in data.get("durl", []) or []:
            url = str(row.get("url", "") or "").strip()
            if url:
                urls.append(url)
        return urls

    def _extract_play_info_urls(self, data: dict[str, Any], pull_protocol: str) -> list[str]:
        urls: list[str] = []
        playurl_info = data.get("playurl_info", {}) or {}
        playurl = playurl_info.get("playurl", {}) or {}
        stream_list = playurl.get("stream", [])

        prefer_flv = pull_protocol == "http_flv"

        for stream in stream_list:
            for fmt in stream.get("format", []) or []:
                format_name = str(fmt.get("format_name", "") or "").lower()
                if prefer_flv and "flv" not in format_name:
                    continue
                if not prefer_flv and "ts" not in format_name and "fmp4" not in format_name:
                    continue
                urls.extend(self._extract_codec_urls(fmt))

        if not urls:
            for stream in stream_list:
                for fmt in stream.get("format", []) or []:
                    urls.extend(self._extract_codec_urls(fmt))

        return urls

    def _extract_codec_urls(self, fmt: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for codec in fmt.get("codec", []) or []:
            base_url = str(codec.get("base_url", "") or "")
            for url_info in codec.get("url_info", []) or []:
                host = str(url_info.get("host", "") or "")
                extra = str(url_info.get("extra", "") or "")
                if host and base_url:
                    urls.append(f"{host}{base_url}{extra}")
        return urls

    def _dedupe_urls(self, urls: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for url in urls:
            if url in seen:
                continue
            deduped.append(url)
            seen.add(url)
        return deduped

    def make_headers(self, cookie: str, room_id: int) -> dict[str, str]:
        headers = {
            "Origin": "https://live.bilibili.com",
            "Referer": f"https://live.bilibili.com/{room_id}",
            "User-Agent": DEFAULT_UA,
        }
        if cookie:
            headers["Cookie"] = cookie
        return headers

from __future__ import annotations

import time
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qsl, urlparse

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


class BiliLoginRequiredError(BiliApiError):
    pass


@dataclass(slots=True)
class RoomPromptMeta:
    room_title: str = ""
    anchor_name: str = ""
    live_status: int | None = None


@dataclass(slots=True)
class BiliLiveSendResult:
    code: int
    message: str
    detail: str = ""


@dataclass(slots=True)
class BiliLoginAccount:
    is_logged_in: bool
    uid: str = ""
    uname: str = ""
    face: str = ""
    source: str = ""
    message: str = ""


@dataclass(slots=True)
class BiliQrLoginSession:
    qrcode_key: str
    url: str
    expires_in_seconds: int = 180


@dataclass(slots=True)
class BiliQrLoginPollResult:
    status: str
    code: int
    message: str
    cookie: str = ""
    refresh_token: str = ""
    account: BiliLoginAccount | None = None
    raw_url: str = ""


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

    async def get_login_account(self, cookie: str = "") -> BiliLoginAccount:
        headers = {
            "User-Agent": DEFAULT_UA,
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
        }
        if cookie:
            headers["Cookie"] = cookie
        async with self._session.get(
            "https://api.bilibili.com/x/web-interface/nav",
            headers=headers,
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if int(payload.get("code", -1)) != 0:
            return BiliLoginAccount(
                is_logged_in=False,
                source="nav",
                message=f"nav failed: code={payload.get('code')} msg={payload.get('message')}",
            )

        data = payload.get("data", {}) or {}
        is_logged_in = bool(data.get("isLogin")) or int(data.get("mid", 0) or 0) > 0
        return BiliLoginAccount(
            is_logged_in=is_logged_in,
            uid=str(data.get("mid", "") or "").strip(),
            uname=str(data.get("uname", "") or "").strip(),
            face=str(data.get("face", "") or "").strip(),
            source="nav",
            message="ok" if is_logged_in else "not_logged_in",
        )

    async def generate_login_qrcode(self) -> BiliQrLoginSession:
        async with self._session.get(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/generate",
            headers={
                "User-Agent": DEFAULT_UA,
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
            },
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)

        if int(payload.get("code", -1)) != 0:
            raise BiliApiError(
                f"qrcode generate failed: code={payload.get('code')} msg={payload.get('message')}"
            )

        data = payload.get("data", {}) or {}
        url = str(data.get("url", "") or "").strip()
        qrcode_key = str(data.get("qrcode_key", "") or "").strip()
        if not url or not qrcode_key:
            raise BiliApiError("qrcode generate returned empty url/qrcode_key")
        return BiliQrLoginSession(qrcode_key=qrcode_key, url=url, expires_in_seconds=180)

    async def poll_login_qrcode(self, qrcode_key: str) -> BiliQrLoginPollResult:
        async with self._session.get(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
            params={"qrcode_key": qrcode_key},
            headers={
                "User-Agent": DEFAULT_UA,
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
            },
            timeout=10,
        ) as resp:
            payload = await resp.json(content_type=None)
            response_headers = resp.headers

        if int(payload.get("code", -1)) != 0:
            raise BiliApiError(
                f"qrcode poll failed: code={payload.get('code')} msg={payload.get('message')}"
            )

        data = payload.get("data", {}) or {}
        raw_code = int(data.get("code", -1))
        raw_message = str(data.get("message", "") or payload.get("message", "") or "").strip()
        raw_url = str(data.get("url", "") or "").strip()
        refresh_token = str(data.get("refresh_token", "") or "").strip()

        status = self._normalize_qr_poll_status(raw_code=raw_code, raw_message=raw_message)
        cookie = ""
        account = None
        if status == "confirmed":
            cookie = self._extract_cookie_from_qr_poll_payload(
                data=data,
                response_headers=response_headers,
            )
            if cookie:
                try:
                    account = await self.get_login_account(cookie)
                except Exception:
                    account = None

        return BiliQrLoginPollResult(
            status=status,
            code=raw_code,
            message=raw_message or status,
            cookie=cookie,
            refresh_token=refresh_token,
            account=account,
            raw_url=raw_url,
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

    async def send_live_danmaku(
        self,
        room_id: int,
        message: str,
        cookie: str = "",
    ) -> BiliLiveSendResult:
        csrf = self._extract_cookie_value(cookie, "bili_jct")
        if not cookie or not csrf:
            raise BiliLoginRequiredError("missing bilibili login cookie or bili_jct csrf token")

        payload = {
            "bubble": 0,
            "msg": str(message or "").strip(),
            "color": 16777215,
            "mode": 1,
            "fontsize": 25,
            "rnd": int(time.time()),
            "roomid": int(room_id or 0),
            "csrf": csrf,
            "csrf_token": csrf,
            "dm_type": 0,
        }
        headers = self.make_headers(cookie=cookie, room_id=room_id)
        headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        async with self._session.post(
            "https://api.live.bilibili.com/msg/send",
            data=payload,
            headers=headers,
            timeout=10,
        ) as resp:
            result = await resp.json(content_type=None)

        code = int(result.get("code", -1))
        message_text = str(result.get("message", "") or result.get("msg", "") or "").strip()
        detail = str(result.get("data", "") or "").strip()
        if code == 0:
            return BiliLiveSendResult(code=code, message=message_text or "ok", detail=detail)
        if code == -101:
            raise BiliLoginRequiredError(f"send live danmaku failed: code={code} msg={message_text}")
        raise BiliApiError(f"send live danmaku failed: code={code} msg={message_text}")

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

    def _normalize_qr_poll_status(self, *, raw_code: int, raw_message: str) -> str:
        mapping = {
            0: "confirmed",
            86038: "expired",
            86090: "waiting_confirm",
            86101: "waiting_scan",
        }
        if raw_code in mapping:
            return mapping[raw_code]
        text = str(raw_message or "").lower()
        if "expired" in text or "失效" in text:
            return "expired"
        if "未确认" in text or "confirm" in text:
            return "waiting_confirm"
        if "未扫码" in text or "scan" in text:
            return "waiting_scan"
        if raw_code == 0:
            return "confirmed"
        return "error"

    def _extract_cookie_from_qr_poll_payload(
        self,
        *,
        data: dict[str, Any],
        response_headers: aiohttp.typedefs.LooseHeaders,
    ) -> str:
        cookie_items: dict[str, str] = {}
        self._merge_cookie_pairs(cookie_items, self._extract_cookie_pairs_from_cookie_info(data))

        for key in ("url", "redirect_url"):
            raw_url = str(data.get(key, "") or "").strip()
            if raw_url:
                self._merge_cookie_pairs(cookie_items, self._extract_cookie_pairs_from_url(raw_url))

        set_cookie_headers: list[str] = []
        if hasattr(response_headers, "getall"):
            try:
                set_cookie_headers = list(response_headers.getall("Set-Cookie", []))
            except Exception:
                set_cookie_headers = []
        self._merge_cookie_pairs(
            cookie_items,
            self._extract_cookie_pairs_from_set_cookie_headers(set_cookie_headers),
        )
        return self._build_cookie_string(cookie_items)

    def _extract_cookie_pairs_from_cookie_info(self, data: dict[str, Any]) -> dict[str, str]:
        cookie_info = data.get("cookie_info", {}) or {}
        raw_cookies = cookie_info.get("cookies", []) or []
        result: dict[str, str] = {}
        for row in raw_cookies:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            value = str(row.get("value", "") or "").strip()
            if name and value:
                result[name] = value
        return result

    def _extract_cookie_pairs_from_url(self, raw_url: str) -> dict[str, str]:
        parsed = urlparse(raw_url)
        items = dict(parse_qsl(parsed.query, keep_blank_values=True))
        if parsed.fragment:
            items.update(dict(parse_qsl(parsed.fragment, keep_blank_values=True)))

        nested_url = str(items.get("gourl", "") or items.get("go_url", "") or "").strip()
        if nested_url:
            items.update(self._extract_cookie_pairs_from_url(nested_url))

        result: dict[str, str] = {}
        for key in (
            "SESSDATA",
            "bili_jct",
            "DedeUserID",
            "DedeUserID__ckMd5",
            "sid",
            "buvid3",
            "buvid4",
            "ac_time_value",
        ):
            value = str(items.get(key, "") or "").strip()
            if value:
                result[key] = value
        return result

    def _extract_cookie_pairs_from_set_cookie_headers(self, set_cookie_headers: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for header in set_cookie_headers:
            jar = SimpleCookie()
            try:
                jar.load(header)
            except Exception:
                continue
            for key, morsel in jar.items():
                value = str(getattr(morsel, "value", "") or "").strip()
                if key and value:
                    result[key] = value
        return result

    def _merge_cookie_pairs(self, target: dict[str, str], source: dict[str, str]) -> None:
        for key, value in source.items():
            if key and value:
                target[key] = value

    def _build_cookie_string(self, cookie_items: dict[str, str]) -> str:
        if not cookie_items:
            return ""
        ordered_keys = [
            "SESSDATA",
            "bili_jct",
            "DedeUserID",
            "DedeUserID__ckMd5",
            "sid",
            "buvid3",
            "buvid4",
            "ac_time_value",
        ]
        parts: list[str] = []
        seen: set[str] = set()
        for key in ordered_keys:
            value = str(cookie_items.get(key, "") or "").strip()
            if value:
                parts.append(f"{key}={value}")
                seen.add(key)
        for key, value in cookie_items.items():
            if key in seen:
                continue
            sval = str(value or "").strip()
            if sval:
                parts.append(f"{key}={sval}")
        return "; ".join(parts)

    def _extract_cookie_value(self, cookie: str, key: str) -> str:
        prefix = f"{key}="
        for part in str(cookie or "").split(";"):
            item = part.strip()
            if item.startswith(prefix):
                return item[len(prefix) :].strip()
        return ""

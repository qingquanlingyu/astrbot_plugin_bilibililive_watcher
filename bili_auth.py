from __future__ import annotations

import json
import hashlib
import inspect
import time
import urllib.parse
from typing import Awaitable, Callable

import aiohttp

WBI_MIXIN_KEY_ENC_TAB = [
    46,
    47,
    18,
    2,
    53,
    8,
    23,
    32,
    15,
    50,
    10,
    31,
    58,
    3,
    45,
    35,
    27,
    43,
    5,
    49,
    33,
    9,
    42,
    19,
    29,
    28,
    14,
    39,
    12,
    38,
    41,
    13,
    37,
    48,
    7,
    16,
    24,
    55,
    40,
    61,
    26,
    17,
    0,
    1,
    60,
    51,
    30,
    4,
    22,
    25,
    54,
    21,
    56,
    59,
    6,
    63,
    57,
    62,
    11,
    36,
    20,
    34,
    44,
    52,
]

MixinKeyGetter = Callable[[], str | Awaitable[str]]
NAV_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
NAV_HEADERS = {
    "User-Agent": NAV_DEFAULT_UA,
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
}


def extract_buvid3(cookie: str) -> str:
    for part in str(cookie or "").split(";"):
        item = part.strip()
        if item.startswith("buvid3="):
            return item
    return ""


class BiliWbiSigner:
    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        *,
        mixin_key_getter: MixinKeyGetter | None = None,
        time_fn: Callable[[], int] | None = None,
    ):
        self._session = session
        self._mixin_key_getter = mixin_key_getter
        self._time_fn = time_fn or (lambda: int(time.time()))
        self._cached_mixin_key = ""

    async def sign(self, params: dict[str, object]) -> dict[str, object]:
        mixin_key = await self.get_mixin_key()
        signed = dict(params)
        signed["wts"] = int(self._time_fn())
        filtered = {
            key: str(value).translate({ord(ch): None for ch in "!'()*"})
            for key, value in signed.items()
        }
        query = urllib.parse.urlencode(sorted(filtered.items()), safe="")
        signed["w_rid"] = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
        return signed

    async def get_mixin_key(self) -> str:
        if self._cached_mixin_key:
            return self._cached_mixin_key

        raw: str
        if self._mixin_key_getter is not None:
            value = self._mixin_key_getter()
            raw = await value if inspect.isawaitable(value) else value
        else:
            if self._session is None:
                raise RuntimeError("WBI signer requires session or explicit mixin key getter")
            async with self._session.get(
                "https://api.bilibili.com/x/web-interface/nav",
                headers=NAV_HEADERS,
                timeout=10,
            ) as resp:
                text = await resp.text()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as e:
                prefix = text[:160].replace("\r", " ").replace("\n", " ")
                raise RuntimeError(
                    f"nav returned non-json response, status={getattr(resp, 'status', '?')} "
                    f"body_prefix={prefix!r}"
                ) from e
            data = payload.get("data", {}) or {}
            wbi_img = data.get("wbi_img", {}) or {}
            raw = (
                str(wbi_img.get("img_url", "")).rsplit("/", 1)[-1].split(".")[0]
                + str(wbi_img.get("sub_url", "")).rsplit("/", 1)[-1].split(".")[0]
            )

        raw = str(raw or "")
        if len(raw) == 32:
            self._cached_mixin_key = raw
            return self._cached_mixin_key

        if len(raw) < max(WBI_MIXIN_KEY_ENC_TAB) + 1:
            raise RuntimeError("invalid WBI mixin key source")

        self._cached_mixin_key = "".join(raw[i] for i in WBI_MIXIN_KEY_ENC_TAB)[:32]
        return self._cached_mixin_key

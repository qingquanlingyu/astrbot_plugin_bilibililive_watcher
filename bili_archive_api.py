from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .bili_auth import extract_cookie_value
else:  # pragma: no cover
    from bili_auth import extract_cookie_value


class BiliArchiveApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        payload: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.payload = dict(payload or {})


@dataclass(slots=True)
class ArchiveSubmitResult:
    aid: str
    bvid: str
    archive_url: str
    payload: dict[str, Any]


class BiliArchiveApi:
    def __init__(self, _session: object | None = None):
        self._session = _session

    async def submit_with_biliup(
        self,
        *,
        cookie: str,
        title: str,
        desc: str,
        tid: int,
        tags: list[str],
        cover_path: str | Path,
        visibility: str,
        video_path: str | Path,
        artifact_dir: str | Path,
    ) -> ArchiveSubmitResult:
        try:
            return await asyncio.to_thread(
                self._submit_with_biliup_sync,
                cookie=cookie,
                title=title,
                desc=desc,
                tid=tid,
                tags=tags,
                cover_path=cover_path,
                visibility=visibility,
                video_path=video_path,
                artifact_dir=artifact_dir,
            )
        except BiliArchiveApiError:
            raise
        except Exception as exc:  # pragma: no cover - third-party runtime paths
            raise BiliArchiveApiError(str(exc), retryable=self._is_retryable_message(str(exc))) from exc

    def _submit_with_biliup_sync(
        self,
        *,
        cookie: str,
        title: str,
        desc: str,
        tid: int,
        tags: list[str],
        cover_path: str | Path,
        visibility: str,
        video_path: str | Path,
        artifact_dir: str | Path,
    ) -> ArchiveSubmitResult:
        BiliBili, Data = self._import_biliup_symbols()
        cookies_payload = self._build_biliup_cookie_payload(cookie)
        artifact_root = Path(artifact_dir).expanduser().resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        cookie_path = artifact_root / "biliup.cookies.json"
        cookie_path.write_text(json.dumps(cookies_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        data = self._build_biliup_video_data(
            Data=Data,
            title=title,
            desc=desc,
            tid=tid,
            tags=tags,
            visibility=visibility,
        )

        video = Path(video_path).expanduser().resolve()
        if not video.exists():
            raise FileNotFoundError(f"video file missing: {video}")
        cover = Path(cover_path).expanduser().resolve()
        if not cover.exists():
            raise FileNotFoundError(f"cover file missing: {cover}")

        try:
            with BiliBili(data) as bili:
                bili.login(str(cookie_path), str(cookie_path))
                data.cover = bili.cover_up(str(cover))
                video_part = bili.upload_file(str(video), lines="AUTO", tasks=3)
                data.append(video_part)
                result = bili.submit("web")
        except Exception as exc:
            message = str(exc or "").strip()
            raise BiliArchiveApiError(message or "biliup submit failed", retryable=self._is_retryable_message(message)) from exc

        payload = result if isinstance(result, dict) else {"raw_result": repr(result)}
        aid = self._pick_result_value(payload, "aid")
        bvid = self._pick_result_value(payload, "bvid")
        archive_url = f"https://www.bilibili.com/video/{bvid}" if bvid else ""
        if not aid and not bvid:
            raise BiliArchiveApiError("biliup submit returned empty aid/bvid", retryable=False, payload=payload)
        return ArchiveSubmitResult(aid=aid, bvid=bvid, archive_url=archive_url, payload=payload)

    def _import_biliup_symbols(self):
        try:
            from biliup.plugins.bili_webup import BiliBili, Data
        except Exception as exc:  # pragma: no cover - depends on installed package
            raise BiliArchiveApiError(
                "biliup dependency unavailable; install requirements.txt first",
                retryable=False,
            ) from exc
        return BiliBili, Data

    def _build_biliup_cookie_payload(self, cookie: str) -> dict[str, Any]:
        raw_cookie = str(cookie or "").strip()
        if not raw_cookie:
            raise BiliArchiveApiError("no effective bilibili cookie available", retryable=False)
        cookies = []
        for key in (
            "SESSDATA",
            "bili_jct",
            "DedeUserID",
            "DedeUserID__ckMd5",
            "sid",
            "ac_time_value",
            "buvid3",
            "buvid4",
            "b_nut",
        ):
            value = extract_cookie_value(raw_cookie, key)
            if value:
                cookies.append({"name": key, "value": value})
        cookie_names = {item["name"] for item in cookies}
        if "SESSDATA" not in cookie_names or "bili_jct" not in cookie_names:
            raise BiliArchiveApiError("missing bilibili login cookie or bili_jct csrf token", retryable=False)
        return {
            "cookie_info": {
                "cookies": cookies,
            },
            "token_info": {
                "access_token": "",
                "refresh_token": "",
            },
        }

    def _build_biliup_video_data(
        self,
        *,
        Data: type,
        title: str,
        desc: str,
        tid: int,
        tags: list[str],
        visibility: str,
    ):
        data = Data()
        data.copyright = 1
        data.source = ""
        data.title = str(title or "").strip()[:80]
        data.desc = str(desc or "").strip()
        data.tid = max(0, int(tid or 0))
        data.dynamic = ""
        data.cover = ""
        data.set_tag(list(tags or []))
        if str(visibility or "").strip().lower() != "public":
            setattr(data, "_visibility_hint", "self_only")
        return data

    def _pick_result_value(self, payload: dict[str, Any], key: str) -> str:
        if key in payload and payload.get(key):
            return str(payload.get(key) or "").strip()
        data = payload.get("data", {}) or {}
        if isinstance(data, dict) and data.get(key):
            return str(data.get(key) or "").strip()
        for nested_key in ("archive", "result"):
            nested = payload.get(nested_key, {}) or {}
            if isinstance(nested, dict) and nested.get(key):
                return str(nested.get(key) or "").strip()
        return ""

    def _is_retryable_message(self, message: str) -> bool:
        lowered = str(message or "").strip().lower()
        return any(
            token in lowered
            for token in (
                "timeout",
                "temporarily",
                "try again",
                "稍后",
                "过快",
                "频繁",
                "风控",
                "captcha",
                "verify",
                "network",
            )
        )

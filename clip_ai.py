from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

if __package__:
    from .clip_review import ClipCandidateStore
    from .clip_time import format_hhmmss, get_session_started_at, parse_hhmmss, wall_ts_to_hhmmss
    from .prompting import DEFAULT_CLIP_REVIEW_PROMPT_TEMPLATE, render_prompt_template
    from .recording_manifest import load_session_index
    from .timeline_store import load_timeline_asr, load_timeline_events
else:  # pragma: no cover
    from clip_review import ClipCandidateStore
    from clip_time import format_hhmmss, get_session_started_at, parse_hhmmss, wall_ts_to_hhmmss
    from prompting import DEFAULT_CLIP_REVIEW_PROMPT_TEMPLATE, render_prompt_template
    from recording_manifest import load_session_index
    from timeline_store import load_timeline_asr, load_timeline_events


class ClipPlannerRuntime:
    def __init__(self, *, session_root: str | Path):
        self.session_root = Path(session_root).expanduser().resolve()
        self.store = ClipCandidateStore(session_root=self.session_root)
        self._session_payload = load_session_index(self.session_root)
        self._session_started_at = get_session_started_at(self.session_root)

    @property
    def session_started_at(self) -> float:
        return self._session_started_at

    @property
    def session_payload(self) -> dict[str, object]:
        return dict(self._session_payload)

    def build_scan_window(
        self,
        *,
        window_seconds: int,
        now_wall_ts: float,
    ) -> dict[str, object]:
        window_end_wall_ts = max(self._session_started_at, float(now_wall_ts or 0.0))
        window_start_wall_ts = max(
            self._session_started_at,
            window_end_wall_ts - max(1, int(window_seconds or 0)),
        )
        asr_rows = [
            item
            for item in load_timeline_asr(self.session_root)
            if float(item.get("wall_ts_end", 0.0) or 0.0) > window_start_wall_ts
            and float(item.get("wall_ts_start", 0.0) or 0.0) < window_end_wall_ts
        ]
        danmaku_rows = [
            item
            for item in load_timeline_events(self.session_root)
            if str(item.get("event_type", "") or "") == "danmaku"
            and window_start_wall_ts
            <= float(item.get("received_wall_ts", 0.0) or 0.0)
            <= window_end_wall_ts
        ]
        ordered_context: list[dict[str, object]] = []
        for item in asr_rows:
            wall_ts_start = float(item.get("wall_ts_start", 0.0) or 0.0)
            wall_ts_end = float(item.get("wall_ts_end", 0.0) or 0.0)
            ordered_context.append(
                {
                    "event_type": "asr",
                    "start_time": wall_ts_to_hhmmss(self.session_root, wall_ts_start),
                    "end_time": wall_ts_to_hhmmss(self.session_root, wall_ts_end),
                    "text": str(item.get("text", "") or "").strip(),
                    "conf": float(item.get("conf", 0.0) or 0.0),
                    "_sort_ts": wall_ts_start,
                }
            )
        for item in danmaku_rows:
            wall_ts = float(item.get("received_wall_ts", 0.0) or 0.0)
            ordered_context.append(
                {
                    "event_type": "danmaku",
                    "time": wall_ts_to_hhmmss(self.session_root, wall_ts),
                    "nickname": str(item.get("nickname", "") or "").strip(),
                    "text": str(item.get("text", "") or "").strip(),
                    "_sort_ts": wall_ts,
                }
            )
        ordered_context.sort(key=lambda item: float(item.get("_sort_ts", 0.0) or 0.0))
        for item in ordered_context:
            item.pop("_sort_ts", None)
        return {
            "window_seconds": max(1, int(window_seconds or 0)),
            "window_start_wall_ts": window_start_wall_ts,
            "window_end_wall_ts": window_end_wall_ts,
            "window_start": wall_ts_to_hhmmss(self.session_root, window_start_wall_ts),
            "window_end": wall_ts_to_hhmmss(self.session_root, window_end_wall_ts),
            "asr_count": len(asr_rows),
            "danmaku_count": len(danmaku_rows),
            "ordered_context": ordered_context,
        }

    def build_scan_prompt(
        self,
        *,
        room_id: int,
        room_title: str,
        anchor_name: str,
        scan_window: dict[str, object],
        prompt_template: str,
    ) -> str:
        context_json = json.dumps(
            {
                "window_seconds": scan_window.get("window_seconds", 0),
                "window_start": scan_window.get("window_start", ""),
                "window_end": scan_window.get("window_end", ""),
                "asr_count": scan_window.get("asr_count", 0),
                "danmaku_count": scan_window.get("danmaku_count", 0),
                "ordered_context": scan_window.get("ordered_context", []),
            },
            ensure_ascii=False,
            indent=2,
        )
        template = str(prompt_template or "").strip() or DEFAULT_CLIP_REVIEW_PROMPT_TEMPLATE
        return render_prompt_template(
            template,
            {
                "anchor_name": anchor_name or "未知主播",
                "room_title": room_title or "(未获取到标题)",
                "room_id": str(room_id or 0),
                "window_seconds": str(scan_window.get("window_seconds", 0) or 0),
                "window_start": str(scan_window.get("window_start", "") or ""),
                "window_end": str(scan_window.get("window_end", "") or ""),
                "context_json": context_json,
            },
        )

    def merge_response_candidate(
        self,
        *,
        response_text: str,
        scan_window: dict[str, object],
    ) -> list[dict]:
        candidate = self._parse_candidate_from_response(
            response_text=response_text,
            scan_window=scan_window,
        )
        if candidate is None:
            return self.store.list_candidates()
        return self.store.merge_candidates([candidate])

    def _parse_candidate_from_response(
        self,
        *,
        response_text: str,
        scan_window: dict[str, object],
    ) -> dict | None:
        raw = str(response_text or "").strip()
        if not raw:
            return None
        normalized = raw.upper()
        if "NO_CLIP" in normalized and not re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", raw):
            return None
        payload = self._extract_json_payload(raw)
        start_text = str(payload.get("start_time", "") or "").strip()
        end_text = str(payload.get("end_time", "") or "").strip()
        topic = str(payload.get("topic", "") or "").strip()
        summary = str(payload.get("summary", "") or "").strip()
        if not start_text or not end_text:
            times = re.findall(r"\b\d{1,2}:\d{2}:\d{2}\b", raw)
            if len(times) >= 2:
                start_text, end_text = times[0], times[1]
        if not start_text or not end_text:
            return None
        try:
            start_wall_ts = self._session_started_at + parse_hhmmss(start_text)
            end_wall_ts = self._session_started_at + parse_hhmmss(end_text)
        except ValueError:
            return None
        window_start_wall_ts = float(scan_window.get("window_start_wall_ts", 0.0) or 0.0)
        window_end_wall_ts = float(scan_window.get("window_end_wall_ts", 0.0) or 0.0)
        if end_wall_ts <= start_wall_ts:
            return None
        if start_wall_ts < (window_start_wall_ts - 1.0) or end_wall_ts > (window_end_wall_ts + 1.0):
            return None
        if not topic:
            topic = "AI 候选片段"
        if not summary:
            summary = raw.splitlines()[0][:80]
        return self._build_candidate(
            clip_start_wall_ts=start_wall_ts,
            clip_end_wall_ts=end_wall_ts,
            score=self._estimate_score(start_wall_ts, end_wall_ts, scan_window),
            topic=topic,
            summary=summary,
            response_text=raw,
            window_seconds=int(scan_window.get("window_seconds", 0) or 0),
        )

    def _build_candidate(
        self,
        *,
        clip_start_wall_ts: float,
        clip_end_wall_ts: float,
        score: float,
        topic: str,
        summary: str,
        response_text: str,
        window_seconds: int,
    ) -> dict:
        candidate_id = self._build_candidate_id(clip_start_wall_ts, clip_end_wall_ts)
        return {
            "candidate_id": candidate_id,
            "candidate_type": "ai_range",
            "state": "pending",
            "room_id": int(self._session_payload.get("room_id", 0) or 0),
            "real_room_id": int(
                self._session_payload.get("real_room_id", self._session_payload.get("room_id", 0)) or 0
            ),
            "anchor_name": str(self._session_payload.get("anchor_name", "") or "").strip(),
            "room_title": str(self._session_payload.get("room_title", "") or "").strip(),
            "session_date": str(self._session_payload.get("session_date", "") or "").strip()
            or self._format_date(self._session_started_at),
            "candidate_date": self._format_date(clip_start_wall_ts),
            "clip_start_wall_ts": round(float(clip_start_wall_ts or 0.0), 3),
            "clip_end_wall_ts": round(float(clip_end_wall_ts or 0.0), 3),
            "score": round(float(score or 0.0), 3),
            "topic": str(topic or "").strip() or "AI 候选片段",
            "summary": str(summary or "").strip(),
            "reasons": [f"window={max(1, int(window_seconds or 0))}s", "source=llm_prompt"],
            "excerpt": str(response_text or "").strip()[:160],
            "created_at": time.time(),
        }

    def _build_candidate_id(self, clip_start_wall_ts: float, clip_end_wall_ts: float) -> str:
        digest = hashlib.sha1(
            f"ai_range|{clip_start_wall_ts:.3f}|{clip_end_wall_ts:.3f}".encode("utf-8")
        ).hexdigest()[:10]
        return f"clip-{digest}"

    def _estimate_score(
        self,
        clip_start_wall_ts: float,
        clip_end_wall_ts: float,
        scan_window: dict[str, object],
    ) -> float:
        duration = max(1.0, clip_end_wall_ts - clip_start_wall_ts)
        asr_count = int(scan_window.get("asr_count", 0) or 0)
        danmaku_count = int(scan_window.get("danmaku_count", 0) or 0)
        return min(99.0, (duration / 30.0) + (asr_count * 0.5) + (danmaku_count * 0.2))

    def _extract_json_payload(self, response_text: str) -> dict[str, object]:
        raw = str(response_text or "").strip()
        if not raw:
            return {}
        for candidate in self._extract_json_candidates(raw):
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _extract_json_candidates(self, response_text: str) -> list[str]:
        matches = re.findall(r"\{[\s\S]*?\}", response_text)
        return [item for item in matches if ":" in item]

    def describe_scan_window(self, scan_window: dict[str, object]) -> str:
        return (
            f"{scan_window.get('window_start', format_hhmmss(0))}"
            f"-{scan_window.get('window_end', format_hhmmss(0))}"
        )

    def _format_date(self, raw_ts: float) -> str:
        if float(raw_ts or 0.0) <= 0:
            return ""
        return time.strftime("%Y-%m-%d", time.localtime(float(raw_ts)))

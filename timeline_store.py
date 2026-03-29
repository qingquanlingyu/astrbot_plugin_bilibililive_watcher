from __future__ import annotations

import json
from pathlib import Path


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def timeline_events_path(session_root: str | Path) -> Path:
    return Path(session_root).expanduser().resolve() / "timeline" / "events.jsonl"


def timeline_asr_path(session_root: str | Path) -> Path:
    return Path(session_root).expanduser().resolve() / "timeline" / "asr.jsonl"


def load_timeline_events(session_root: str | Path) -> list[dict]:
    return _load_jsonl(timeline_events_path(session_root))


def load_timeline_asr(session_root: str | Path) -> list[dict]:
    return _load_jsonl(timeline_asr_path(session_root))


def query_asr_range(
    session_root: str | Path,
    *,
    start_wall_ts: float,
    end_wall_ts: float,
) -> list[dict]:
    start = float(start_wall_ts or 0.0)
    end = float(end_wall_ts or 0.0)
    if end <= start:
        return []
    rows = []
    for item in load_timeline_asr(session_root):
        item_start = float(item.get("wall_ts_start", 0.0) or 0.0)
        item_end = float(item.get("wall_ts_end", 0.0) or 0.0)
        if item_end <= start or item_start >= end:
            continue
        rows.append(item)
    return sorted(rows, key=lambda item: float(item.get("wall_ts_start", 0.0) or 0.0))


class TimelineIndexerRuntime:
    def __init__(self, *, session_root: str | Path, session_id: str):
        self.session_root = Path(session_root).expanduser().resolve()
        self.session_id = str(session_id or "").strip() or self.session_root.name
        self.events_path = timeline_events_path(self.session_root)
        self.asr_path = timeline_asr_path(self.session_root)

    def append_danmaku(
        self,
        *,
        uid: str,
        nickname: str,
        text: str,
        timeline: str,
        source: str,
        received_wall_ts: float,
    ) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "event_type": "danmaku",
            "source": source,
            "uid": uid,
            "nickname": nickname,
            "text": text,
            "timeline": timeline,
            "received_wall_ts": float(received_wall_ts or 0.0),
        }
        _append_jsonl(self.events_path, payload)
        return payload

    def append_asr(
        self,
        *,
        text: str,
        ts_start: float,
        ts_end: float,
        wall_ts_start: float,
        wall_ts_end: float,
        conf: float,
        source: str,
    ) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "event_type": "asr",
            "source": source,
            "text": text,
            "ts_start": float(ts_start or 0.0),
            "ts_end": float(ts_end or 0.0),
            "wall_ts_start": float(wall_ts_start or 0.0),
            "wall_ts_end": float(wall_ts_end or 0.0),
            "conf": float(conf or 0.0),
        }
        _append_jsonl(self.events_path, payload)
        _append_jsonl(self.asr_path, payload)
        return payload

    def append_marker(
        self,
        *,
        marker_id: str,
        wall_ts: float,
        label: str,
        source: str = "manual",
    ) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "event_type": "marker",
            "source": source,
            "marker_id": marker_id,
            "wall_ts": float(wall_ts or 0.0),
            "label": str(label or "").strip(),
        }
        _append_jsonl(self.events_path, payload)
        return payload

    def append_recording_segment(
        self,
        *,
        segment_id: str,
        file_path: str,
        wall_ts_start: float,
        wall_ts_end: float,
        ok: bool,
        error: str = "",
    ) -> dict[str, object]:
        payload = {
            "session_id": self.session_id,
            "event_type": "recording_segment",
            "source": "recording",
            "segment_id": segment_id,
            "file_path": file_path,
            "wall_ts_start": float(wall_ts_start or 0.0),
            "wall_ts_end": float(wall_ts_end or 0.0),
            "ok": bool(ok),
            "error": str(error or "").strip(),
        }
        _append_jsonl(self.events_path, payload)
        return payload

    def flush(self) -> None:
        return None

from __future__ import annotations

import re
from pathlib import Path

if __package__:
    from .recording_manifest import load_session_index
else:  # pragma: no cover
    from recording_manifest import load_session_index


_HHMMSS_RE = re.compile(r"^\s*(\d{1,2}):([0-5]\d):([0-5]\d)\s*$")


def parse_hhmmss(text: str) -> int:
    raw = str(text or "").strip()
    match = _HHMMSS_RE.match(raw)
    if not match:
        raise ValueError(f"invalid HH:MM:SS: {raw}")
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return (hours * 3600) + (minutes * 60) + seconds


def format_hhmmss(total_seconds: float | int) -> str:
    value = max(0, int(float(total_seconds or 0.0)))
    hours = value // 3600
    value -= hours * 3600
    minutes = value // 60
    seconds = value - (minutes * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_session_started_at(session_root: str | Path) -> float:
    payload = load_session_index(session_root)
    started_at = float(payload.get("started_at", 0.0) or 0.0)
    if started_at <= 0:
        raise ValueError("session started_at is unavailable")
    return started_at


def hhmmss_to_wall_ts(session_root: str | Path, value: str) -> float:
    return get_session_started_at(session_root) + parse_hhmmss(value)


def wall_ts_to_hhmmss(session_root: str | Path, wall_ts: float) -> str:
    started_at = get_session_started_at(session_root)
    return format_hhmmss(max(0.0, float(wall_ts or 0.0) - started_at))


def resolve_range_to_wall_ts(session_root: str | Path, start_text: str, end_text: str) -> tuple[float, float]:
    start_wall_ts = hhmmss_to_wall_ts(session_root, start_text)
    end_wall_ts = hhmmss_to_wall_ts(session_root, end_text)
    if end_wall_ts <= start_wall_ts:
        raise ValueError("end time must be greater than start time")
    return start_wall_ts, end_wall_ts

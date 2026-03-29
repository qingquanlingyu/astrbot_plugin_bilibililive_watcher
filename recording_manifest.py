from __future__ import annotations

import json
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SessionLayout:
    root: Path
    segments_dir: Path
    timeline_dir: Path
    clips_dir: Path
    reports_dir: Path


@dataclass(slots=True)
class RecordingSegment:
    segment_id: str
    file_path: str
    wall_ts_start: float
    wall_ts_end: float
    duration_seconds: float
    stream_url_fingerprint: str
    ok: bool
    returncode: int
    error: str = ""


@dataclass(slots=True)
class RecordingSessionMeta:
    room_id: int
    real_room_id: int
    session_id: str
    session_dir: str
    started_at: float
    ended_at: float = 0.0
    notes: list[str] | None = None
    segments: list[dict] | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["notes"] = list(self.notes or [])
        payload["segments"] = list(self.segments or [])
        return payload


def build_session_id(started_at: float) -> str:
    return time.strftime("session-%Y%m%d-%H%M%S", time.localtime(started_at))


def build_session_layout(output_root: str | Path, room_id: int, started_at: float) -> SessionLayout:
    root = Path(output_root).expanduser().resolve() / "recordings" / str(room_id) / build_session_id(started_at)
    segments_dir = root / "segments"
    timeline_dir = root / "timeline"
    clips_dir = root / "clips"
    reports_dir = root / "reports"
    for path in (segments_dir, timeline_dir, clips_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return SessionLayout(
        root=root,
        segments_dir=segments_dir,
        timeline_dir=timeline_dir,
        clips_dir=clips_dir,
        reports_dir=reports_dir,
    )


def session_manifest_path(session_root: str | Path) -> Path:
    return Path(session_root).expanduser().resolve() / "session_manifest.json"


def load_session_index(session_root: str | Path) -> dict[str, object]:
    path = session_manifest_path(session_root)
    if not path.exists():
        return {"segments": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_session_index(session_root: str | Path, payload: dict[str, object]) -> dict[str, object]:
    path = session_manifest_path(session_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def init_session_index(
    session_root: str | Path,
    *,
    room_id: int,
    real_room_id: int,
    started_at: float,
    session_id: str = "",
    notes: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    root = Path(session_root).expanduser().resolve()
    payload = RecordingSessionMeta(
        room_id=room_id,
        real_room_id=real_room_id,
        session_id=session_id or root.name,
        session_dir=str(root),
        started_at=started_at,
        notes=list(notes or []),
        segments=[],
    ).to_dict()
    if extra:
        payload.update(extra)
    return save_session_index(root, payload)


def append_segment_index(session_root: str | Path, segment: RecordingSegment) -> dict[str, object]:
    payload = load_session_index(session_root)
    segments = payload.setdefault("segments", [])
    if not isinstance(segments, list):
        segments = []
        payload["segments"] = segments
    segments.append(asdict(segment))
    return save_session_index(session_root, payload)


def update_session_index(session_root: str | Path, **fields: object) -> dict[str, object]:
    payload = load_session_index(session_root)
    payload.update(fields)
    return save_session_index(session_root, payload)

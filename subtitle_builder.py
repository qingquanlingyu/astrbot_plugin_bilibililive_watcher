from __future__ import annotations

from pathlib import Path

if __package__:
    from .clip_exporter import build_srt_text
    from .timeline_store import load_timeline_asr
else:  # pragma: no cover
    from clip_exporter import build_srt_text
    from timeline_store import load_timeline_asr


def build_subtitle_for_clip(
    *,
    session_root: str | Path,
    clip_start_wall_ts: float,
    clip_end_wall_ts: float,
    output_path: str | Path,
) -> str:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(clip_end_wall_ts or 0.0) - float(clip_start_wall_ts or 0.0))
    srt_text = build_srt_text(
        load_timeline_asr(session_root),
        clip_start_wall_ts=float(clip_start_wall_ts or 0.0),
        clip_duration_seconds=duration,
    )
    output.write_text(srt_text, encoding="utf-8")
    return str(output)

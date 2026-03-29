from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .clip_exporter import resolve_clip_session_root
    from .clip_review import ClipCandidateStore
    from .clip_time import wall_ts_to_hhmmss
else:  # pragma: no cover
    from clip_exporter import resolve_clip_session_root
    from clip_review import ClipCandidateStore
    from clip_time import wall_ts_to_hhmmss


@dataclass(slots=True)
class PublishDraft:
    clip_id: str
    session_id: str
    session_root: str
    clip_output_path: str
    clip_duration_seconds: float
    cover_local_path: str
    cover_remote_url: str
    title: str
    desc: str
    tags: list[str]
    tid: int
    visibility: str
    source_candidate_id: str


def build_publish_draft(
    *,
    clip_row: dict[str, Any],
    title_template: str,
    desc_template: str,
    default_tid: int,
    default_tags: list[str],
    visibility: str,
    explicit_title: str = "",
    explicit_desc: str = "",
    explicit_tags: list[str] | None = None,
    explicit_tid: int | None = None,
) -> PublishDraft:
    session_root = resolve_clip_session_root(clip_row)
    candidate = _find_source_candidate(session_root=session_root, clip_row=clip_row)
    source_candidate_id = str(candidate.get("candidate_id", "") or "").strip()
    clip_range = _build_clip_range(session_root, clip_row)
    clip_date = _resolve_clip_date(clip_row)
    room_id = _resolve_room_id(clip_row)
    room_url = f"https://live.bilibili.com/{room_id}" if room_id > 0 else ""
    title = resolve_title(
        explicit_title=explicit_title,
        clip_row=clip_row,
        candidate_row=candidate,
        title_template=title_template,
        clip_range=clip_range,
        clip_date=clip_date,
    )
    desc = resolve_desc(
        explicit_desc=explicit_desc,
        clip_row=clip_row,
        candidate_row=candidate,
        desc_template=desc_template,
        clip_range=clip_range,
        clip_date=clip_date,
        room_url=room_url,
    )
    tags = _clean_tags(explicit_tags if explicit_tags is not None else default_tags)
    tid = max(0, int(explicit_tid if explicit_tid is not None else default_tid))
    return PublishDraft(
        clip_id=str(clip_row.get("clip_id", "") or "").strip(),
        session_id=str(clip_row.get("session_id", "") or "").strip() or session_root.name,
        session_root=str(session_root),
        clip_output_path=str(clip_row.get("output_path", "") or "").strip(),
        clip_duration_seconds=max(0.0, float(clip_row.get("duration_seconds", 0.0) or 0.0)),
        cover_local_path="",
        cover_remote_url="",
        title=title,
        desc=desc,
        tags=tags,
        tid=tid,
        visibility=_normalize_visibility(visibility),
        source_candidate_id=source_candidate_id,
    )


def resolve_title(
    *,
    explicit_title: str,
    clip_row: dict[str, Any],
    candidate_row: dict[str, Any],
    title_template: str,
    clip_range: str,
    clip_date: str,
) -> str:
    title = str(explicit_title or "").strip()
    if title:
        return title[:80]
    label = str(clip_row.get("label", "") or "").strip()
    if label:
        return label[:80]
    topic = str(candidate_row.get("topic", "") or "").strip()
    if topic:
        return topic[:80]
    room_title = str(clip_row.get("room_title", "") or "").strip()
    anchor_name = str(clip_row.get("anchor_name", "") or "").strip()
    rendered = _render_template(
        title_template or "{{room_title}} {{clip_range}} 切片",
        {
            "anchor_name": anchor_name,
            "room_title": room_title,
            "clip_range": clip_range,
            "clip_date": clip_date,
            "clip_id": str(clip_row.get("clip_id", "") or "").strip(),
        },
    ).strip()
    if rendered:
        return rendered[:80]
    base = anchor_name or room_title or "直播片段"
    return f"{base} {clip_range} 切片".strip()[:80]


def resolve_desc(
    *,
    explicit_desc: str,
    clip_row: dict[str, Any],
    candidate_row: dict[str, Any],
    desc_template: str,
    clip_range: str,
    clip_date: str,
    room_url: str,
) -> str:
    anchor_name = str(clip_row.get("anchor_name", "") or "").strip()
    prefix = "\n".join(
        [
            f"主播：{anchor_name}" if anchor_name else "主播：",
            f"直播间：{room_url}" if room_url else "直播间：",
            f"日期：{clip_date}" if clip_date else "日期：",
        ]
    ).rstrip()
    if explicit_desc.strip():
        body = explicit_desc.strip()
    else:
        auto_desc = _build_auto_desc(clip_row=clip_row, candidate_row=candidate_row, clip_range=clip_range)
        rendered = _render_template(
            desc_template
            or (
                "主播：{{anchor_name}}\n"
                "直播间：https://live.bilibili.com/{{real_room_id}}\n"
                "日期：{{clip_date}}\n\n"
                "{{auto_desc}}"
            ),
            {
                "anchor_name": anchor_name,
                "real_room_id": str(_resolve_room_id(clip_row) or ""),
                "clip_date": clip_date,
                "clip_range": clip_range,
                "room_title": str(clip_row.get("room_title", "") or "").strip(),
                "auto_desc": auto_desc,
                "room_url": room_url,
            },
        ).strip()
        body = rendered
    cleaned = body.strip()
    if cleaned.startswith(prefix):
        return cleaned
    if not cleaned:
        return prefix
    return f"{prefix}\n\n{cleaned}"


def _find_source_candidate(session_root: Path, clip_row: dict[str, Any]) -> dict[str, Any]:
    candidate_store = ClipCandidateStore(session_root=session_root)
    rows = candidate_store.load_candidates()
    source = str(clip_row.get("source", "") or "").strip()
    clip_id = str(clip_row.get("clip_id", "") or "").strip()
    source_candidate_id = ""
    if source.startswith("candidate:"):
        source_candidate_id = source.split(":", 1)[1].strip()
    if source_candidate_id:
        for row in rows:
            if str(row.get("candidate_id", "") or "").strip() == source_candidate_id:
                return dict(row)
    for row in rows:
        if str(row.get("exported_clip_id", "") or "").strip() == clip_id:
            return dict(row)
    return {}


def _build_clip_range(session_root: Path, clip_row: dict[str, Any]) -> str:
    try:
        start = wall_ts_to_hhmmss(session_root, float(clip_row.get("clip_start_wall_ts", 0.0) or 0.0))
        end = wall_ts_to_hhmmss(session_root, float(clip_row.get("clip_end_wall_ts", 0.0) or 0.0))
        return f"{start}-{end}"
    except Exception:
        return "00:00:00-00:00:00"


def _resolve_clip_date(clip_row: dict[str, Any]) -> str:
    clip_date = str(clip_row.get("clip_date", "") or "").strip()
    if clip_date:
        return clip_date
    session_date = str(clip_row.get("session_date", "") or "").strip()
    if session_date:
        return session_date
    wall_ts = float(clip_row.get("clip_start_wall_ts", 0.0) or 0.0)
    if wall_ts > 0:
        import time

        return time.strftime("%Y-%m-%d", time.localtime(wall_ts))
    return ""


def _resolve_room_id(clip_row: dict[str, Any]) -> int:
    for key in ("real_room_id", "room_id"):
        try:
            value = int(clip_row.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    return 0


def _build_auto_desc(*, clip_row: dict[str, Any], candidate_row: dict[str, Any], clip_range: str) -> str:
    lines: list[str] = []
    room_title = str(clip_row.get("room_title", "") or "").strip()
    if room_title:
        lines.append(f"直播标题：{room_title}")
    if clip_range:
        lines.append(f"片段时间：{clip_range}")
    summary = str(candidate_row.get("summary", "") or "").strip()
    if summary:
        lines.append(f"片段摘要：{summary}")
    topic = str(candidate_row.get("topic", "") or "").strip()
    if topic and topic not in " ".join(lines):
        lines.append(f"片段主题：{topic}")
    return "\n".join(lines).strip()


def _render_template(template: str, mapping: dict[str, str]) -> str:
    rendered = str(template or "")
    for key, value in mapping.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value or ""))
    return rendered


def _clean_tags(raw_tags: list[str] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in list(raw_tags or []):
        text = str(tag or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text[:20])
    return cleaned[:12]


def _normalize_visibility(value: str) -> str:
    return "self_only" if str(value or "").strip().lower() != "public" else "public"

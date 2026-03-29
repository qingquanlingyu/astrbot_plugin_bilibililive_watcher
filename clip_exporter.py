from __future__ import annotations

import asyncio
import json
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .recording_manifest import load_session_index
    from .timeline_store import load_timeline_asr
else:  # pragma: no cover
    from recording_manifest import load_session_index
    from timeline_store import load_timeline_asr


@dataclass(slots=True)
class ClipRange:
    start_wall_ts: float
    end_wall_ts: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_wall_ts - self.start_wall_ts)


@dataclass(slots=True)
class ClipManifest:
    clip_id: str
    session_id: str
    room_id: int
    real_room_id: int
    anchor_name: str
    room_title: str
    session_date: str
    clip_date: str
    source: str
    clip_start_wall_ts: float
    clip_end_wall_ts: float
    duration_seconds: float
    output_path: str
    segment_ids: list[str]
    created_at: float
    srt_path: str = ""
    marker_id: str = ""
    label: str = ""


def _format_srt_timestamp(raw_seconds: float) -> str:
    total_ms = max(0, int(round(raw_seconds * 1000)))
    hours = total_ms // 3600000
    total_ms -= hours * 3600000
    minutes = total_ms // 60000
    total_ms -= minutes * 60000
    seconds = total_ms // 1000
    milliseconds = total_ms - (seconds * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def plan_segments_for_clip(
    segments: list[dict],
    clip_start_wall_ts: float,
    clip_duration_seconds: float,
) -> list[dict]:
    clip_end_wall_ts = clip_start_wall_ts + clip_duration_seconds
    ordered = sorted(
        [dict(item) for item in segments],
        key=lambda item: float(item.get("wall_ts_start", 0.0) or 0.0),
    )
    if not ordered:
        raise RuntimeError("no recorded segments found in session manifest")
    earliest_start = float(ordered[0].get("wall_ts_start", 0.0) or 0.0)
    latest_end = max(float(item.get("wall_ts_end", 0.0) or 0.0) for item in ordered)
    if clip_start_wall_ts < earliest_start or clip_end_wall_ts > latest_end:
        raise ValueError("clip range is outside recorded segment coverage")

    epsilon = 0.001
    current = float(clip_start_wall_ts or 0.0)
    plan: list[dict] = []
    for segment in ordered:
        segment_start = float(segment.get("wall_ts_start", 0.0) or 0.0)
        segment_end = float(segment.get("wall_ts_end", 0.0) or 0.0)
        if segment_end <= current + epsilon:
            continue
        if segment_start > current + epsilon:
            raise ValueError("clip range crosses an unrecorded gap between segments")
        overlap_start = max(current, segment_start)
        overlap_end = min(clip_end_wall_ts, segment_end)
        if overlap_end <= overlap_start:
            continue
        item = dict(segment)
        item["clip_part_start_wall_ts"] = overlap_start
        item["clip_part_end_wall_ts"] = overlap_end
        item["clip_part_offset_seconds"] = max(0.0, overlap_start - segment_start)
        item["clip_part_duration_seconds"] = max(0.0, overlap_end - overlap_start)
        plan.append(item)
        current = overlap_end
        if current >= clip_end_wall_ts - epsilon:
            break
    if current < clip_end_wall_ts - epsilon:
        raise ValueError("clip range crosses an unrecorded gap between segments")
    return plan


def build_srt_text(
    asr_events: list[dict],
    clip_start_wall_ts: float,
    clip_duration_seconds: float,
) -> str:
    clip_end_wall_ts = clip_start_wall_ts + clip_duration_seconds
    cues: list[str] = []
    idx = 1
    for event in asr_events:
        text = str(event.get("text", "") or "").strip()
        start = float(event.get("wall_ts_start", 0.0) or 0.0)
        end = float(event.get("wall_ts_end", 0.0) or 0.0)
        if not text or end <= clip_start_wall_ts or start >= clip_end_wall_ts:
            continue
        rel_start = max(0.0, start - clip_start_wall_ts)
        rel_end = min(clip_duration_seconds, end - clip_start_wall_ts)
        if rel_end <= rel_start:
            continue
        cues.extend(
            [
                str(idx),
                f"{_format_srt_timestamp(rel_start)} --> {_format_srt_timestamp(rel_end)}",
                text,
                "",
            ]
        )
        idx += 1
    return "\n".join(cues).strip() + ("\n" if cues else "")


def load_clip_manifest(session_root: str | Path) -> list[dict]:
    path = Path(session_root).expanduser().resolve() / "clips" / "clip_manifest.jsonl"
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def save_clip_manifest_row(session_root: str | Path, row: dict[str, object]) -> dict[str, object]:
    rows = load_clip_manifest(session_root)
    target_id = str(row.get("clip_id", "") or "").strip()
    replaced = False
    for index, current in enumerate(rows):
        if str(current.get("clip_id", "") or "").strip() != target_id:
            continue
        rows[index] = dict(row)
        replaced = True
        break
    if not replaced:
        rows.append(dict(row))
    path = Path(session_root).expanduser().resolve() / "clips" / "clip_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return dict(row)


def resolve_clip_session_root(clip_row: dict[str, object]) -> Path:
    raw_session_root = str(clip_row.get("session_root", "") or "").strip()
    if raw_session_root:
        return Path(raw_session_root).expanduser().resolve()
    output_path = str(clip_row.get("output_path", "") or "").strip()
    if output_path:
        candidate = Path(output_path).expanduser().resolve().parent.parent
        if candidate.exists():
            return candidate
    raise ValueError("clip manifest row is missing a usable session_root/output_path")


def find_clip_by_id(storage_root: str | Path, clip_id: str) -> dict[str, object] | None:
    target_id = str(clip_id or "").strip()
    if not target_id:
        return None
    recordings_root = Path(storage_root).expanduser().resolve() / "recordings"
    if not recordings_root.exists():
        return None
    for manifest_path in sorted(recordings_root.rglob("clip_manifest.jsonl")):
        try:
            session_root = manifest_path.parent.parent
            for row in load_clip_manifest(session_root):
                if str(row.get("clip_id", "") or "").strip() != target_id:
                    continue
                resolved = dict(row)
                resolved["session_root"] = str(session_root)
                return resolved
        except Exception:
            continue
    return None


def _format_date(raw_ts: float) -> str:
    if float(raw_ts or 0.0) <= 0:
        return ""
    return time.strftime("%Y-%m-%d", time.localtime(float(raw_ts)))


class ClipExporterRuntime:
    def __init__(
        self,
        *,
        session_root: str | Path,
        ffmpeg_path: str = "ffmpeg",
        output_container: str = "mp4",
    ):
        self.session_root = Path(session_root).expanduser().resolve()
        self.ffmpeg_path = str(ffmpeg_path or "ffmpeg").strip() or "ffmpeg"
        self.output_container = str(output_container or "mp4").strip() or "mp4"
        self.clips_dir = self.session_root / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.clip_manifest_path = self.clips_dir / "clip_manifest.jsonl"

    async def export_last_seconds(
        self,
        *,
        seconds: float,
        source: str,
        build_srt: bool,
        label: str = "",
        marker_id: str = "",
    ) -> dict[str, object]:
        payload = load_session_index(self.session_root)
        segments = self._ok_segments(payload)
        if not segments:
            raise RuntimeError("no recorded segments found in session manifest")
        latest_end = max(float(segment.get("wall_ts_end", 0.0) or 0.0) for segment in segments)
        earliest_start = min(float(segment.get("wall_ts_start", 0.0) or 0.0) for segment in segments)
        clip_end_wall_ts = latest_end
        clip_start_wall_ts = max(earliest_start, clip_end_wall_ts - max(1.0, float(seconds or 0.0)))
        return await self.export_clip(
            clip_range=ClipRange(clip_start_wall_ts, clip_end_wall_ts),
            source=source,
            build_srt=build_srt,
            label=label,
            marker_id=marker_id,
        )

    async def export_around_timestamp(
        self,
        *,
        center_wall_ts: float,
        pre_seconds: float,
        post_seconds: float,
        source: str,
        build_srt: bool,
        label: str = "",
        marker_id: str = "",
    ) -> dict[str, object]:
        clip_range = ClipRange(
            start_wall_ts=max(0.0, float(center_wall_ts or 0.0) - max(0.0, float(pre_seconds or 0.0))),
            end_wall_ts=max(0.0, float(center_wall_ts or 0.0) + max(0.0, float(post_seconds or 0.0))),
        )
        return await self.export_clip(
            clip_range=clip_range,
            source=source,
            build_srt=build_srt,
            label=label,
            marker_id=marker_id,
        )

    async def export_clip(
        self,
        *,
        clip_range: ClipRange,
        source: str,
        build_srt: bool,
        label: str = "",
        marker_id: str = "",
        clip_id: str = "",
    ) -> dict[str, object]:
        payload = load_session_index(self.session_root)
        session_id = str(payload.get("session_id", "") or self.session_root.name)
        room_id = int(payload.get("room_id", 0) or 0)
        real_room_id = int(payload.get("real_room_id", room_id) or room_id)
        anchor_name = str(payload.get("anchor_name", "") or "").strip()
        room_title = str(payload.get("room_title", "") or "").strip()
        session_date = str(payload.get("session_date", "") or "").strip() or _format_date(
            float(payload.get("started_at", 0.0) or 0.0)
        )
        clip_date = _format_date(clip_range.start_wall_ts)
        segments = self._ok_segments(payload)
        if not segments:
            raise RuntimeError("no recorded segments found in session manifest")
        if clip_range.duration_seconds <= 0:
            raise ValueError("clip duration must be positive")
        segment_plan = plan_segments_for_clip(
            segments,
            clip_start_wall_ts=clip_range.start_wall_ts,
            clip_duration_seconds=clip_range.duration_seconds,
        )
        resolved_clip_id = str(clip_id or "").strip() or time.strftime("clip-%Y%m%d-%H%M%S", time.localtime())
        output_path = self.clips_dir / f"{resolved_clip_id}.{self.output_container}"
        if len(segment_plan) == 1:
            target_segment = segment_plan[0]
            source_path = Path(str(target_segment.get("file_path", "") or "")).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"segment file missing: {source_path}")
            await self._run_ffmpeg_clip(
                input_path=source_path,
                output_path=output_path,
                start_seconds=float(target_segment.get("clip_part_offset_seconds", 0.0) or 0.0),
                duration_seconds=float(target_segment.get("clip_part_duration_seconds", 0.0) or 0.0),
            )
        else:
            await self._run_ffmpeg_clip_across_segments(
                segment_plan=segment_plan,
                output_path=output_path,
            )
        srt_path = ""
        if build_srt:
            srt_text = build_srt_text(
                load_timeline_asr(self.session_root),
                clip_start_wall_ts=clip_range.start_wall_ts,
                clip_duration_seconds=clip_range.duration_seconds,
            )
            srt_output = output_path.with_suffix(".srt")
            srt_output.write_text(srt_text, encoding="utf-8")
            srt_path = str(srt_output)
        manifest = ClipManifest(
            clip_id=resolved_clip_id,
            session_id=session_id,
            room_id=room_id,
            real_room_id=real_room_id,
            anchor_name=anchor_name,
            room_title=room_title,
            session_date=session_date,
            clip_date=clip_date,
            source=source,
            clip_start_wall_ts=clip_range.start_wall_ts,
            clip_end_wall_ts=clip_range.end_wall_ts,
            duration_seconds=clip_range.duration_seconds,
            output_path=str(output_path),
            segment_ids=[str(item.get("segment_id", "") or "") for item in segment_plan],
            created_at=time.time(),
            srt_path=srt_path,
            marker_id=marker_id,
            label=label,
        )
        return save_clip_manifest_row(self.session_root, asdict(manifest))

    def _ok_segments(self, payload: dict[str, object]) -> list[dict]:
        segments = list(payload.get("segments", []) or [])
        return [segment for segment in segments if bool(segment.get("ok", True))]

    async def _run_ffmpeg_clip_across_segments(
        self,
        *,
        segment_plan: list[dict],
        output_path: Path,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="biliwatch-clip-", dir=str(self.clips_dir)) as temp_dir:
            temp_root = Path(temp_dir)
            part_paths: list[Path] = []
            for index, item in enumerate(segment_plan, start=1):
                source_path = Path(str(item.get("file_path", "") or "")).expanduser().resolve()
                if not source_path.exists():
                    raise FileNotFoundError(f"segment file missing: {source_path}")
                part_path = temp_root / f"part-{index:03d}.mkv"
                await self._run_ffmpeg_clip(
                    input_path=source_path,
                    output_path=part_path,
                    start_seconds=float(item.get("clip_part_offset_seconds", 0.0) or 0.0),
                    duration_seconds=float(item.get("clip_part_duration_seconds", 0.0) or 0.0),
                )
                part_paths.append(part_path)
            await self._concat_clip_parts(
                part_paths=part_paths,
                output_path=output_path,
            )

    async def _concat_clip_parts(
        self,
        *,
        part_paths: list[Path],
        output_path: Path,
    ) -> None:
        if not part_paths:
            raise RuntimeError("no temporary clip parts to concatenate")
        list_path = output_path.parent / f".{output_path.stem}.concat.txt"
        try:
            lines: list[str] = []
            for path in part_paths:
                escaped = str(path).replace("'", "'\\''")
                lines.append(f"file '{escaped}'\n")
            list_path.write_text(
                "".join(lines),
                encoding="utf-8",
            )
            await self._run_ffmpeg_concat(
                list_path=list_path,
                output_path=output_path,
                reencode=False,
            )
        except Exception:
            await self._run_ffmpeg_concat(
                list_path=list_path,
                output_path=output_path,
                reencode=True,
            )
        finally:
            if list_path.exists():
                list_path.unlink(missing_ok=True)

    async def _run_ffmpeg_clip(
        self,
        *,
        input_path: Path,
        output_path: Path,
        start_seconds: float,
        duration_seconds: float,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-c",
            "copy",
            str(output_path),
        ]
        await self._run_ffmpeg_command(cmd, output_path=output_path)

    async def _run_ffmpeg_concat(
        self,
        *,
        list_path: Path,
        output_path: Path,
        reencode: bool,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
        ]
        if reencode:
            cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac"])
        else:
            cmd.extend(["-c", "copy"])
        cmd.append(str(output_path))
        await self._run_ffmpeg_command(cmd, output_path=output_path)

    async def _run_ffmpeg_command(self, cmd: list[str], *, output_path: Path) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_lines: list[str] = []
        assert proc.stderr is not None
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            stderr_lines.append(line)
            if len(stderr_lines) > 5:
                stderr_lines = stderr_lines[-5:]
        returncode = await proc.wait()
        if returncode != 0 or not output_path.exists():
            raise RuntimeError(f"ffmpeg clip export failed rc={returncode} err={' | '.join(stderr_lines[-3:])}")

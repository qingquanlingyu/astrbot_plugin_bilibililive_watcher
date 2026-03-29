from __future__ import annotations

import asyncio
from pathlib import Path


async def extract_cover_frame(
    *,
    clip_path: str | Path,
    output_path: str | Path,
    ffmpeg_path: str = "ffmpeg",
    seek_seconds: float = 0.0,
) -> str:
    input_path = Path(clip_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"clip file missing: {input_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path or "ffmpeg").strip() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, float(seek_seconds or 0.0)):.3f}",
        "-i",
        str(input_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(target_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    assert proc.stderr is not None
    stderr_lines: list[str] = []
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
    if returncode != 0 or not target_path.exists():
        raise RuntimeError(
            f"ffmpeg cover extraction failed rc={returncode} err={' | '.join(stderr_lines[-3:])}"
        )
    return str(target_path)

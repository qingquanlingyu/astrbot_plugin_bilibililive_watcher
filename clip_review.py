from __future__ import annotations

import json
import time
from pathlib import Path


def _candidate_manifest_path(session_root: str | Path) -> Path:
    return Path(session_root).expanduser().resolve() / "clips" / "candidate_manifest.json"


class ClipCandidateStore:
    def __init__(self, *, session_root: str | Path):
        self.session_root = Path(session_root).expanduser().resolve()
        self.path = _candidate_manifest_path(self.session_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_candidates(self) -> list[dict]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        rows = payload.get("candidates", [])
        return list(rows) if isinstance(rows, list) else []

    def save_candidates(self, candidates: list[dict]) -> list[dict]:
        payload = {"candidates": list(candidates or [])}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload["candidates"]

    def merge_candidates(self, candidates: list[dict]) -> list[dict]:
        existing = self.load_candidates()
        by_id = {str(item.get("candidate_id", "") or ""): dict(item) for item in existing}
        now = time.time()
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", "") or "").strip()
            if not candidate_id:
                continue
            current = by_id.get(candidate_id)
            if current is None:
                merged = dict(candidate)
                merged.setdefault("state", "pending")
                merged.setdefault("created_at", now)
                merged["updated_at"] = now
                by_id[candidate_id] = merged
                continue
            preserved_state = str(current.get("state", "pending") or "pending")
            preserved_exported_clip_id = str(current.get("exported_clip_id", "") or "")
            preserved_exported_output_path = str(current.get("exported_output_path", "") or "")
            merged = dict(current)
            merged.update(candidate)
            merged["state"] = preserved_state
            merged["exported_clip_id"] = preserved_exported_clip_id
            merged["exported_output_path"] = preserved_exported_output_path
            merged["updated_at"] = now
            by_id[candidate_id] = merged
        merged_candidates = sorted(
            by_id.values(),
            key=lambda item: (
                -float(item.get("score", 0.0) or 0.0),
                float(item.get("clip_start_wall_ts", 0.0) or 0.0),
            ),
        )
        return self.save_candidates(merged_candidates)

    def list_candidates(self, *, state: str = "") -> list[dict]:
        rows = self.load_candidates()
        normalized = str(state or "").strip().lower()
        if not normalized:
            return rows
        return [item for item in rows if str(item.get("state", "") or "").strip().lower() == normalized]

    def get_candidate(self, candidate_id: str) -> dict:
        target = str(candidate_id or "").strip()
        for item in self.load_candidates():
            if str(item.get("candidate_id", "") or "") == target:
                return item
        raise KeyError(f"candidate not found: {target}")

    def update_state(self, candidate_id: str, state: str) -> dict:
        return self.update_fields(candidate_id, state=str(state or "").strip().lower() or "pending")

    def update_fields(self, candidate_id: str, **fields: object) -> dict:
        target = str(candidate_id or "").strip()
        rows = self.load_candidates()
        updated: dict | None = None
        for item in rows:
            if str(item.get("candidate_id", "") or "") != target:
                continue
            item.update(fields)
            item["updated_at"] = time.time()
            updated = item
            break
        if updated is None:
            raise KeyError(f"candidate not found: {target}")
        self.save_candidates(rows)
        return updated

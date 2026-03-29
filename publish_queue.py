from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import field
from dataclasses import MISSING
from pathlib import Path
from typing import Any
from typing import Awaitable
from typing import Callable

import aiohttp

if __package__:
    from .bili_archive_api import (
        ArchiveSubmitResult,
        BiliArchiveApi,
        BiliArchiveApiError,
    )
    from .publish_cover import extract_cover_frame
    from .publish_metadata import PublishDraft
else:  # pragma: no cover
    from bili_archive_api import (
        ArchiveSubmitResult,
        BiliArchiveApi,
        BiliArchiveApiError,
    )
    from publish_cover import extract_cover_frame
    from publish_metadata import PublishDraft


PUBLISH_RUNNING_STATES = {"queued", "preparing", "uploading_cover", "uploading_video", "submitting"}
PUBLISH_EDITABLE_STATES = {"draft"}
PUBLISH_UNFINISHED_STATES = PUBLISH_EDITABLE_STATES | PUBLISH_RUNNING_STATES | {"retry_waiting"}
PUBLISH_TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
PUBLISH_DUPLICATE_BLOCKING_STATES = PUBLISH_UNFINISHED_STATES | {"succeeded"}


@dataclass(slots=True)
class PublishJob:
    job_id: str
    state: str
    clip_id: str
    session_id: str
    session_root: str
    clip_output_path: str
    clip_duration_seconds: float
    cover_local_path: str = ""
    cover_remote_url: str = ""
    title: str = ""
    desc: str = ""
    tags: list[str] = field(default_factory=list)
    tid: int = 0
    visibility: str = "self_only"
    source_candidate_id: str = ""
    retry_count: int = 0
    max_retries: int = 3
    retry_backoff_seconds: int = 300
    next_retry_at: float = 0.0
    last_error: str = ""
    aid: str = ""
    bvid: str = ""
    archive_url: str = ""
    final_request_summary: dict[str, Any] = field(default_factory=dict)
    use_tid_predict: bool = True
    use_tag_recommendation: bool = True
    cover_strategy: str = "midpoint_frame"
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublishJob:
        data = {}
        raw = dict(payload or {})
        for item in dataclass_fields(cls):
            if item.name in raw:
                data[item.name] = raw[item.name]
                continue
            if item.default is not MISSING:
                data[item.name] = item.default
            elif item.default_factory is not MISSING:
                data[item.name] = item.default_factory()
        data["tags"] = list(data.get("tags", []) or [])
        data["final_request_summary"] = dict(data.get("final_request_summary", {}) or {})
        return cls(**data)


class PublishJobStore:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        sanitize_error_message: Callable[[Exception | str], str] | None = None,
    ):
        self.storage_root = Path(storage_root).expanduser().resolve()
        self.publish_root = self.storage_root / "publish"
        self.publish_root.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.publish_root / "jobs.json"
        self.events_path = self.publish_root / "events.jsonl"
        self.artifacts_root = self.publish_root / "artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self._sanitize_error_message = sanitize_error_message or (lambda err: str(err or "").strip())

    def list_jobs(self) -> list[PublishJob]:
        if not self.jobs_path.exists():
            return []
        payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        rows = payload.get("jobs", []) if isinstance(payload, dict) else []
        jobs: list[PublishJob] = []
        for row in rows if isinstance(rows, list) else []:
            try:
                jobs.append(PublishJob.from_dict(dict(row)))
            except Exception:
                continue
        return sorted(jobs, key=lambda item: (float(item.created_at or 0.0), item.job_id), reverse=True)

    def get_job(self, job_id: str) -> PublishJob | None:
        target = str(job_id or "").strip()
        for job in self.list_jobs():
            if job.job_id == target:
                return job
        return None

    def save_jobs(self, jobs: list[PublishJob]) -> list[PublishJob]:
        ordered = sorted(jobs, key=lambda item: (float(item.created_at or 0.0), item.job_id), reverse=True)
        payload = {"jobs": [item.to_dict() for item in ordered]}
        self.jobs_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ordered

    def upsert_job(self, job: PublishJob) -> PublishJob:
        jobs = self.list_jobs()
        replaced = False
        for index, current in enumerate(jobs):
            if current.job_id != job.job_id:
                continue
            jobs[index] = job
            replaced = True
            break
        if not replaced:
            jobs.append(job)
        self.save_jobs(jobs)
        return job

    def update_job(self, job_id: str, **fields: Any) -> PublishJob:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(f"publish job not found: {job_id}")
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        job.updated_at = time.time()
        self.upsert_job(job)
        return job

    def append_event(self, job_id: str, event_type: str, **fields: Any) -> None:
        payload = {
            "ts": time.time(),
            "job_id": str(job_id or "").strip(),
            "event": str(event_type or "").strip(),
        }
        for key, value in fields.items():
            if key == "error":
                payload[key] = self._sanitize_error_message(value)
            elif isinstance(value, dict):
                payload[key] = dict(value)
            elif isinstance(value, list):
                payload[key] = list(value)
            else:
                payload[key] = value
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def artifact_dir(self, job_id: str) -> Path:
        path = self.artifacts_root / str(job_id or "").strip()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def find_duplicate(self, clip_id: str) -> PublishJob | None:
        target = str(clip_id or "").strip()
        for job in self.list_jobs():
            if job.clip_id == target and job.state in PUBLISH_DUPLICATE_BLOCKING_STATES:
                return job
        return None

    def recover_jobs(self) -> list[PublishJob]:
        jobs = self.list_jobs()
        changed = False
        now = time.time()
        for job in jobs:
            if job.state in {"preparing", "uploading_cover", "uploading_video", "submitting"}:
                previous_state = job.state
                job.state = "queued"
                job.updated_at = now
                changed = True
                self.append_event(job.job_id, "job_recovered", from_state=previous_state, to_state="queued")
            elif job.state == "queued":
                continue
            elif job.state == "retry_waiting" and job.next_retry_at <= 0:
                job.next_retry_at = now + max(1, int(job.retry_backoff_seconds or 0))
                job.updated_at = now
                changed = True
        if changed:
            self.save_jobs(jobs)
        return jobs


class PublishRuntime:
    def __init__(
        self,
        *,
        storage_root: str | Path,
        session: aiohttp.ClientSession,
        enabled_getter: Callable[[], bool],
        cookie_getter: Callable[[], str],
        ffmpeg_path_getter: Callable[[], str],
        sanitize_error_message: Callable[[Exception | str], str] | None = None,
        archive_api_factory: Callable[[aiohttp.ClientSession], BiliArchiveApi] | None = None,
        cover_extractor: Callable[..., Awaitable[str]] | None = None,
    ):
        self.store = PublishJobStore(
            storage_root=storage_root,
            sanitize_error_message=sanitize_error_message,
        )
        self._session = session
        self._enabled_getter = enabled_getter
        self._cookie_getter = cookie_getter
        self._ffmpeg_path_getter = ffmpeg_path_getter
        self._sanitize_error_message = sanitize_error_message or (lambda err: str(err or "").strip())
        self._archive_api = (
            archive_api_factory(session) if archive_api_factory is not None else BiliArchiveApi(session)
        )
        self._cover_extractor = cover_extractor or extract_cover_frame
        self._task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.store.recover_jobs()
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._worker_loop(), name="bili-publish-runtime")
        self._wake_event.set()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None

    async def submit(
        self,
        draft: PublishDraft,
        *,
        max_retries: int,
        retry_backoff_seconds: int,
        use_tid_predict: bool,
        use_tag_recommendation: bool,
        cover_strategy: str,
    ) -> tuple[PublishJob, PublishJob | None]:
        async with self._lock:
            duplicate = self.store.find_duplicate(draft.clip_id)
            if duplicate is not None:
                return duplicate, duplicate
            now = time.time()
            job = PublishJob(
                job_id=_build_job_id(now),
                state="draft",
                clip_id=draft.clip_id,
                session_id=draft.session_id,
                session_root=draft.session_root,
                clip_output_path=draft.clip_output_path,
                clip_duration_seconds=draft.clip_duration_seconds,
                cover_local_path=draft.cover_local_path,
                cover_remote_url=draft.cover_remote_url,
                title=draft.title,
                desc=draft.desc,
                tags=list(draft.tags or []),
                tid=max(0, int(draft.tid or 0)),
                visibility=str(draft.visibility or "self_only").strip() or "self_only",
                source_candidate_id=draft.source_candidate_id,
                retry_count=0,
                max_retries=max(0, int(max_retries or 0)),
                retry_backoff_seconds=max(1, int(retry_backoff_seconds or 0)),
                use_tid_predict=bool(use_tid_predict),
                use_tag_recommendation=bool(use_tag_recommendation),
                cover_strategy=str(cover_strategy or "midpoint_frame").strip() or "midpoint_frame",
                created_at=now,
                updated_at=now,
            )
            self.store.upsert_job(job)
            self.store.append_event(
                job.job_id,
                "job_drafted",
                clip_id=job.clip_id,
                title=job.title,
                visibility=job.visibility,
            )
        return job, None

    async def update_draft(self, job_id: str, **fields: Any) -> PublishJob:
        async with self._lock:
            job = self.store.get_job(job_id)
            if job is None:
                raise KeyError(f"publish job not found: {job_id}")
            if job.state not in PUBLISH_EDITABLE_STATES:
                raise ValueError(f"job state does not allow editing: {job.state}")
            if "title" in fields:
                fields["title"] = str(fields.get("title", "") or "").strip()[:80]
            if "desc" in fields:
                fields["desc"] = str(fields.get("desc", "") or "").strip()
            if "tags" in fields:
                tags = []
                seen: set[str] = set()
                for tag in list(fields.get("tags", []) or []):
                    text = str(tag or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    tags.append(text[:20])
                fields["tags"] = tags[:12]
            if "tid" in fields:
                fields["tid"] = max(0, int(fields.get("tid", 0) or 0))
            job = self.store.update_job(job_id, **fields)
            self.store.append_event(job.job_id, "job_draft_updated", fields=list(fields.keys()))
        return job

    async def approve(self, job_id: str) -> PublishJob:
        async with self._lock:
            job = self.store.get_job(job_id)
            if job is None:
                raise KeyError(f"publish job not found: {job_id}")
            if job.state != "draft":
                raise ValueError(f"job state does not allow approve: {job.state}")
            job = self.store.update_job(job_id, state="queued", next_retry_at=0.0, last_error="")
            self.store.append_event(job.job_id, "job_enqueued", clip_id=job.clip_id, title=job.title)
        self._wake_event.set()
        return job

    async def retry(self, job_id: str) -> PublishJob:
        async with self._lock:
            job = self.store.get_job(job_id)
            if job is None:
                raise KeyError(f"publish job not found: {job_id}")
            if job.state not in {"retry_waiting", "failed"}:
                raise ValueError(f"job state does not allow retry: {job.state}")
            job.state = "queued"
            job.next_retry_at = 0.0
            job.last_error = ""
            job.updated_at = time.time()
            self.store.upsert_job(job)
            self.store.append_event(job.job_id, "job_retry_enqueued", retry_count=job.retry_count)
        self._wake_event.set()
        return job

    def get_job(self, job_id: str) -> PublishJob | None:
        return self.store.get_job(job_id)

    def list_jobs(self, *, limit: int = 10) -> list[PublishJob]:
        return self.store.list_jobs()[: max(1, int(limit or 10))]

    async def _worker_loop(self) -> None:
        while True:
            if not self._enabled_getter():
                await self._wait_for_work(timeout_seconds=5.0)
                continue
            job = self._pick_next_job()
            if job is None:
                await self._wait_for_work(timeout_seconds=self._next_retry_delay())
                continue
            await self._run_job(job)

    async def _wait_for_work(self, *, timeout_seconds: float) -> None:
        timeout = max(0.5, float(timeout_seconds or 0.0))
        self._wake_event.clear()
        try:
            await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
        except TimeoutError:
            return

    def _pick_next_job(self) -> PublishJob | None:
        now = time.time()
        waiting: list[PublishJob] = []
        for job in self.store.list_jobs():
            if job.state == "queued":
                waiting.append(job)
            elif job.state == "retry_waiting" and job.next_retry_at <= now:
                waiting.append(job)
        if not waiting:
            return None
        waiting.sort(key=lambda item: (float(item.created_at or 0.0), item.job_id))
        return waiting[0]

    def _next_retry_delay(self) -> float:
        future_times = [
            float(job.next_retry_at or 0.0)
            for job in self.store.list_jobs()
            if job.state == "retry_waiting" and float(job.next_retry_at or 0.0) > time.time()
        ]
        if not future_times:
            return 30.0
        return max(0.5, min(future_times) - time.time())

    async def _run_job(self, job: PublishJob) -> None:
        try:
            await self._process_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_job_failure(job, exc)

    async def _process_job(self, job: PublishJob) -> None:
        cookie = str(self._cookie_getter() or "").strip()
        if not cookie:
            raise BiliArchiveApiError("no effective bilibili cookie available", retryable=False)
        clip_path = Path(job.clip_output_path).expanduser().resolve()
        if not clip_path.exists():
            raise FileNotFoundError(f"clip file missing: {clip_path}")
        job = self.store.update_job(job.job_id, state="preparing", last_error="")
        if not job.cover_local_path:
            artifact_dir = self.store.artifact_dir(job.job_id)
            cover_path = artifact_dir / "cover.jpg"
            seek_seconds = max(0.0, float(job.clip_duration_seconds or 0.0) / 2.0)
            local_cover = await self._cover_extractor(
                clip_path=clip_path,
                output_path=cover_path,
                ffmpeg_path=self._ffmpeg_path_getter(),
                seek_seconds=seek_seconds,
            )
            job = self.store.update_job(job.job_id, cover_local_path=str(local_cover))
            self.store.append_event(job.job_id, "cover_generated", cover_local_path=str(local_cover))
        job = self.store.update_job(job.job_id, state="uploading_video")
        self.store.append_event(
            job.job_id,
            "video_upload_started",
            filename=clip_path.name,
            backend="biliup",
        )
        job = self.store.update_job(job.job_id, state="submitting")
        submit_result = await self._archive_api.submit_with_biliup(
            cookie=cookie,
            title=job.title,
            desc=job.desc,
            tid=job.tid,
            tags=list(job.tags or []),
            cover_path=job.cover_local_path,
            visibility=job.visibility,
            video_path=clip_path,
            artifact_dir=self.store.artifact_dir(job.job_id),
        )
        summary = self._build_request_summary(job=job, result=submit_result, video_filename=clip_path.name)
        self.store.update_job(
            job.job_id,
            state="succeeded",
            aid=submit_result.aid,
            bvid=submit_result.bvid,
            archive_url=submit_result.archive_url,
            final_request_summary=summary,
            last_error="",
            next_retry_at=0.0,
        )
        self.store.append_event(
            job.job_id,
            "archive_submitted",
            aid=submit_result.aid,
            bvid=submit_result.bvid,
            archive_url=submit_result.archive_url,
        )
        self.store.append_event(job.job_id, "job_succeeded", aid=submit_result.aid, bvid=submit_result.bvid)

    async def _handle_job_failure(self, job: PublishJob, exc: Exception) -> None:
        current = self.store.get_job(job.job_id) or job
        error_text = self._sanitize_error_message(exc)
        retryable = bool(getattr(exc, "retryable", False))
        if retryable and current.retry_count < current.max_retries:
            next_retry_at = time.time() + max(1, int(current.retry_backoff_seconds or 0))
            self.store.update_job(
                current.job_id,
                state="retry_waiting",
                retry_count=current.retry_count + 1,
                next_retry_at=next_retry_at,
                last_error=error_text,
            )
            self.store.append_event(
                current.job_id,
                "job_retry_scheduled",
                retry_count=current.retry_count + 1,
                next_retry_at=next_retry_at,
                error=error_text,
            )
            self._wake_event.set()
            return
        self.store.update_job(
            current.job_id,
            state="failed",
            last_error=error_text,
            next_retry_at=0.0,
        )
        self.store.append_event(current.job_id, "job_failed", error=error_text, retryable=retryable)

    def _build_request_summary(
        self,
        *,
        job: PublishJob,
        result: ArchiveSubmitResult,
        video_filename: str,
    ) -> dict[str, Any]:
        return {
            "title": job.title,
            "tid": job.tid,
            "tags": list(job.tags or []),
            "visibility": job.visibility,
            "cover_local_path": job.cover_local_path,
            "video_filename": video_filename,
            "aid": result.aid,
            "bvid": result.bvid,
            "archive_url": result.archive_url,
        }


def _build_job_id(now_ts: float) -> str:
    return f"pub-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now_ts))}-{secrets.token_hex(3)}"

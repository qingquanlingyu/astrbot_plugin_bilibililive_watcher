from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import aiohttp

from bili_archive_api import BiliArchiveApi
from clip_exporter import find_clip_by_id
from publish_metadata import build_publish_draft
from publish_queue import PublishRuntime


class _FakeArchiveApi:
    async def submit_with_biliup(
        self,
        *,
        cookie: str,
        title: str,
        desc: str,
        tid: int,
        tags: list[str],
        cover_path: str | Path,
        visibility: str,
        video_path: str | Path,
        artifact_dir: str | Path,
    ):
        from bili_archive_api import ArchiveSubmitResult

        return ArchiveSubmitResult(
            aid="10001",
            bvid="BV1fake1",
            archive_url="https://www.bilibili.com/video/BV1fake1",
            payload={"code": 0},
        )


async def _fake_cover_extractor(*, clip_path: str | Path, output_path: str | Path, ffmpeg_path: str, seek_seconds: float):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")
    return str(path)


class PublishWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="biliwatch-publish-test-")
        self.root = Path(self.temp_dir.name)

    async def asyncTearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_recording_fixture(self) -> Path:
        session_root = self.root / "recordings" / "123" / "session-20260329-120000"
        clips_dir = session_root / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / "clip-20260329-120100.mp4"
        clip_path.write_bytes(b"fake-video")
        (session_root / "session_manifest.json").write_text(
            json.dumps(
                {
                    "session_id": "session-20260329-120000",
                    "started_at": 1000,
                    "room_id": 123,
                    "real_room_id": 456,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        with (clips_dir / "clip_manifest.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "clip_id": "clip-20260329-120100",
                        "session_id": "session-20260329-120000",
                        "room_id": 123,
                        "real_room_id": 456,
                        "anchor_name": "测试主播",
                        "room_title": "测试直播间",
                        "session_date": "2026-03-29",
                        "clip_date": "2026-03-29",
                        "source": "candidate:hot-1",
                        "clip_start_wall_ts": 1010,
                        "clip_end_wall_ts": 1050,
                        "duration_seconds": 40.0,
                        "output_path": str(clip_path),
                        "segment_ids": [],
                        "created_at": 12345,
                        "label": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        (clips_dir / "candidate_manifest.json").write_text(
            json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": "hot-1",
                            "topic": "高能反应",
                            "summary": "主播打出关键操作",
                            "exported_clip_id": "clip-20260329-120100",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return session_root

    async def test_find_clip_and_build_publish_draft(self):
        self._build_recording_fixture()
        clip_row = find_clip_by_id(self.root, "clip-20260329-120100")
        self.assertIsNotNone(clip_row)
        assert clip_row is not None
        draft = build_publish_draft(
            clip_row=clip_row,
            title_template="{{room_title}} {{clip_range}} 切片",
            desc_template=(
                "主播：{{anchor_name}}\n"
                "直播间：https://live.bilibili.com/{{real_room_id}}\n"
                "日期：{{clip_date}}\n\n"
                "{{auto_desc}}"
            ),
            default_tid=0,
            default_tags=["默认标签"],
            visibility="self_only",
        )
        self.assertEqual(draft.clip_id, "clip-20260329-120100")
        self.assertEqual(draft.source_candidate_id, "hot-1")
        self.assertEqual(draft.visibility, "self_only")
        self.assertEqual(draft.title, "高能反应")
        self.assertTrue(draft.desc.startswith("主播：测试主播\n直播间：https://live.bilibili.com/456\n日期：2026-03-29"))

    async def test_biliup_cookie_payload_shape(self):
        api = BiliArchiveApi()
        payload = api._build_biliup_cookie_payload(
            "SESSDATA=test_sess;bili_jct=test_csrf;DedeUserID=1"
        )
        self.assertIn("cookie_info", payload)
        self.assertIn("token_info", payload)
        self.assertEqual(payload["token_info"]["access_token"], "")
        cookies = payload["cookie_info"]["cookies"]
        self.assertTrue(any(item["name"] == "SESSDATA" for item in cookies))
        self.assertTrue(any(item["name"] == "bili_jct" for item in cookies))

    async def test_biliup_video_data_shape(self):
        from biliup.plugins.bili_webup import Data

        api = BiliArchiveApi()
        data = api._build_biliup_video_data(
            Data=Data,
            title="测试标题",
            desc="测试简介",
            tid=17,
            tags=["切片", "直播"],
            visibility="self_only",
        )
        self.assertEqual(data.copyright, 1)
        self.assertEqual(data.source, "")
        self.assertEqual(data.tag, "切片,直播")
        self.assertEqual(data.tid, 17)
        self.assertEqual(data.title, "测试标题")

    async def test_publish_runtime_processes_job(self):
        self._build_recording_fixture()
        clip_row = find_clip_by_id(self.root, "clip-20260329-120100")
        assert clip_row is not None
        draft = build_publish_draft(
            clip_row=clip_row,
            title_template="{{room_title}} {{clip_range}} 切片",
            desc_template="{{auto_desc}}",
            default_tid=0,
            default_tags=[],
            visibility="self_only",
        )
        session = aiohttp.ClientSession()
        runtime = PublishRuntime(
            storage_root=self.root,
            session=session,
            enabled_getter=lambda: True,
            cookie_getter=lambda: "SESSDATA=test;bili_jct=csrf-token",
            ffmpeg_path_getter=lambda: "ffmpeg",
            sanitize_error_message=lambda err: str(err),
            archive_api_factory=lambda _session: _FakeArchiveApi(),
            cover_extractor=_fake_cover_extractor,
        )
        try:
            await runtime.start()
            job, duplicate = await runtime.submit(
                draft,
                max_retries=2,
                retry_backoff_seconds=1,
                use_tid_predict=True,
                use_tag_recommendation=True,
                cover_strategy="midpoint_frame",
            )
            self.assertIsNone(duplicate)
            current = runtime.get_job(job.job_id)
            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(current.state, "draft")
            current = await runtime.update_draft(job.job_id, desc="人工修改后的简介", tags=["人工标签"])
            self.assertEqual(current.desc, "人工修改后的简介")
            self.assertEqual(current.tags, ["人工标签"])
            current = await runtime.approve(job.job_id)
            self.assertEqual(current.state, "queued")
            for _ in range(40):
                current = runtime.get_job(job.job_id)
                if current is not None and current.state == "succeeded":
                    break
                await asyncio.sleep(0.05)
            current = runtime.get_job(job.job_id)
            self.assertIsNotNone(current)
            assert current is not None
            self.assertEqual(current.state, "succeeded")
            self.assertEqual(current.bvid, "BV1fake1")
            self.assertEqual(current.tid, 0)
            self.assertEqual(current.tags, ["人工标签"])
        finally:
            await runtime.stop()
            await session.close()


if __name__ == "__main__":
    unittest.main()

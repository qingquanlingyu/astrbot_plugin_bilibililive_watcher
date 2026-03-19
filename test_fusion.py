from __future__ import annotations

import unittest

from fusion import FusionEngine
from models import DanmakuItem


class FusionEngineSingerModeTests(unittest.TestCase):
    def setUp(self):
        self.engine = FusionEngine()

    def _danmaku(self, *, text: str, ts: float) -> DanmakuItem:
        return DanmakuItem(
            uid="1",
            nickname="观众",
            text=text,
            ts=ts,
            timeline="00:00",
            dedup_key=f"{ts}:{text}",
        )

    def test_singer_mode_triggers_when_keyword_hits_within_window(self):
        summary = self.engine.build_summary(
            danmaku_items=[
                self._danmaku(text="普通聊天", ts=100.0),
                self._danmaku(text="这段真的好听", ts=118.0),
            ],
            asr_segments=[],
            window_seconds=45,
            singer_mode_enabled=True,
            singer_mode_keywords=["好听", "打call"],
            singer_mode_window_seconds=20,
        )

        self.assertEqual(summary.scene_mode, "singer")
        self.assertEqual(summary.singer_hit_keywords, ["好听"])
        self.assertEqual(summary.singer_window_seconds, 20)

    def test_singer_mode_ignores_keyword_outside_window(self):
        summary = self.engine.build_summary(
            danmaku_items=[
                self._danmaku(text="好听", ts=10.0),
                self._danmaku(text="现在在聊天", ts=40.0),
            ],
            asr_segments=[],
            window_seconds=45,
            singer_mode_enabled=True,
            singer_mode_keywords=["好听"],
            singer_mode_window_seconds=20,
        )

        self.assertEqual(summary.scene_mode, "chat")
        self.assertEqual(summary.singer_hit_keywords, [])


if __name__ == "__main__":
    unittest.main()

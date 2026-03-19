from __future__ import annotations

import unittest

from models import FusionSummary
from prompting import build_fused_prompt


class PromptingTests(unittest.TestCase):
    def test_prompt_includes_current_bili_nickname_and_self_repeat_rule(self):
        fusion = FusionSummary(
            window_seconds=45,
            danmaku_count=2,
            ordered_context=[
                {"source": "弹幕", "speaker": "测试号", "text": "来了"},
                {"source": "主播", "speaker": "主播", "text": "谢谢大家"},
            ],
        )

        prompt = build_fused_prompt(
            room_id=123,
            room_title="测试直播间",
            anchor_name="测试主播",
            self_bili_nickname="测试号",
            fusion=fusion,
            max_reply_chars=20,
        )

        self.assertIn("你当前在 B 站直播间使用的昵称：测试号", prompt)
        self.assertIn("speaker 与你当前 B 站昵称相同", prompt)
        self.assertIn("那就是你自己之前发过的弹幕", prompt)


if __name__ == "__main__":
    unittest.main()

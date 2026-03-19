from __future__ import annotations

import unittest

from models import FusionSummary
from prompting import build_fused_prompt, render_prompt_template


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

    def test_prompt_template_renders_dynamic_variables_and_computed_reply_limit(self):
        fusion = FusionSummary(
            window_seconds=30,
            danmaku_count=1,
            singer_window_seconds=20,
            ordered_context=[{"source": "弹幕", "speaker": "观众A", "text": "好听"}],
        )

        prompt = build_fused_prompt(
            room_id=7788,
            room_title="晚间歌会",
            anchor_name="主播A",
            self_bili_nickname="路人甲",
            fusion=fusion,
            max_reply_chars=3,
            prompt_template=(
                "主播={{anchor_name}}\n"
                "标题={{room_title}}\n"
                "昵称={{self_bili_nickname}}\n"
                "上限={{reply_length_limit}}\n"
                "模式={{scene_mode}}\n"
                "上下文={{context_json}}"
            ),
        )

        self.assertIn("主播=主播A", prompt)
        self.assertIn("标题=晚间歌会", prompt)
        self.assertIn("昵称=路人甲", prompt)
        self.assertIn("上限=10", prompt)
        self.assertIn("模式=chat", prompt)
        self.assertIn('"room_id": 7788', prompt)
        self.assertIn('"speaker": "观众A"', prompt)

    def test_render_prompt_template_replaces_unknown_variables_with_empty_string(self):
        rendered = render_prompt_template("A={{known}} B={{unknown}}", {"known": "1"})

        self.assertEqual(rendered, "A=1 B=")

    def test_singer_mode_instruction_is_configurable_and_supports_template_variables(self):
        fusion = FusionSummary(
            window_seconds=30,
            danmaku_count=2,
            scene_mode="singer",
            singer_hit_keywords=["好听"],
            singer_window_seconds=18,
            ordered_context=[{"source": "弹幕", "speaker": "观众A", "text": "好听"}],
        )

        prompt = build_fused_prompt(
            room_id=10086,
            room_title="深夜歌会",
            anchor_name="主播B",
            self_bili_nickname="路人甲",
            fusion=fusion,
            max_reply_chars=18,
            singer_mode_instruction="6) {{anchor_name}} 正在唱歌，只允许夸唱功，不要接歌词。",
        )

        self.assertIn("6) 主播B 正在唱歌，只允许夸唱功，不要接歌词。", prompt)


if __name__ == "__main__":
    unittest.main()

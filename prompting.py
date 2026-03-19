from __future__ import annotations

import json
import re

try:  # pragma: no cover
    from .models import FusionSummary
except ImportError:  # pragma: no cover
    from models import FusionSummary

DEFAULT_SINGER_MODE_INSTRUCTION = "6) 当前是唱歌场景，可以参考弹幕发送“好听”或“打call”，严禁根据主播歌词回复。"

DEFAULT_FUSED_PROMPT_TEMPLATE = """下面是 B 站直播间中的主播语音转写与观众弹幕信息。
主播昵称：{{anchor_name}}
直播间标题：{{room_title}}
直播间房间号：{{room_id}}
你当前在 B 站直播间使用的昵称：{{self_bili_nickname}}
请基于这些信息，按照你当前的人设，生成 1 条适合发送到直播间的互动弹幕。
输出要求：
1) 只输出一句话，不要解释，不要 Markdown。
2) 长度不超过 {{reply_length_limit}} 字，口语自然。
3) 不要过分强调人设，不要攻击性，模仿已有弹幕风格。
4) 优先参考 ordered_context 中按时间排序后的事件序列，理解弹幕与语音内容的先后关系，越近的越优先。
5) 如果 ordered_context 中某条弹幕的 speaker 与你当前 B 站昵称相同，那就是你自己之前发过的弹幕；不要把它作为风格参考。
{{singer_mode_instruction}}

ordered_context:
{{context_json}}
"""


def build_fused_prompt(
    *,
    room_id: int,
    room_title: str,
    anchor_name: str,
    self_bili_nickname: str,
    fusion: FusionSummary,
    max_reply_chars: int,
    prompt_template: str = "",
    singer_mode_instruction: str = DEFAULT_SINGER_MODE_INSTRUCTION,
) -> str:
    payload = {
        "window_seconds": fusion.window_seconds,
        "room_id": room_id,
        "scene": {
            "mode": fusion.scene_mode,
            "constraints": fusion.constraints,
            "singer_window_seconds": fusion.singer_window_seconds,
            "singer_hit_keywords": fusion.singer_hit_keywords,
        },
        "ordered_context": fusion.ordered_context,
    }
    context_json = json.dumps(payload, ensure_ascii=False, indent=2)
    template = str(prompt_template or "").strip() or DEFAULT_FUSED_PROMPT_TEMPLATE
    variables = {
        "anchor_name": anchor_name or "未知主播",
        "room_title": room_title or "(未获取到标题)",
        "room_id": str(room_id),
        "self_bili_nickname": self_bili_nickname or "(未知，无法识别自己的历史弹幕)",
        "max_reply_chars": str(max_reply_chars),
        "reply_length_limit": str(max(10, max_reply_chars)),
        "context_json": context_json,
        "scene_mode": fusion.scene_mode,
        "scene_constraints_json": json.dumps(fusion.constraints, ensure_ascii=False),
        "singer_window_seconds": str(fusion.singer_window_seconds),
        "singer_hit_keywords_json": json.dumps(fusion.singer_hit_keywords, ensure_ascii=False),
    }
    variables["singer_mode_instruction"] = (
        render_prompt_template(str(singer_mode_instruction or "").strip(), variables)
        if fusion.scene_mode == "singer"
        else ""
    )
    return render_prompt_template(template, variables)


def render_prompt_template(template: str, variables: dict[str, str]) -> str:
    pattern = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(variables.get(key, ""))

    rendered = pattern.sub(_replace, str(template or ""))
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered

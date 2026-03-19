from __future__ import annotations

import json

try:  # pragma: no cover
    from .models import FusionSummary
except ImportError:  # pragma: no cover
    from models import FusionSummary


def build_fused_prompt(
    *,
    room_id: int,
    room_title: str,
    anchor_name: str,
    self_bili_nickname: str,
    fusion: FusionSummary,
    max_reply_chars: int,
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
    lines = [
        "下面是 B 站直播间中的主播语音转写与观众弹幕信息。",
        f"主播昵称：{anchor_name or '未知主播'}",
        f"直播间标题：{room_title or '(未获取到标题)'}",
        f"直播间房间号：{room_id}",
        f"你当前在 B 站直播间使用的昵称：{self_bili_nickname or '(未知，无法识别自己的历史弹幕)'}",
        "请基于这些信息，按照你当前的人设，生成 1 条适合发送到直播间的互动弹幕。",
        "输出要求：",
        "1) 只输出一句话，不要解释，不要 Markdown。",
        f"2) 长度不超过 {max(10, max_reply_chars)} 字，口语自然。",
        "3) 不要过分强调人设，不要攻击性，模仿已有弹幕风格。",
        "4) 优先参考 ordered_context 中按时间排序后的事件序列，理解弹幕与语音内容的先后关系，越近的越优先。",
        "5) 如果 ordered_context 中某条弹幕的 speaker 与你当前 B 站昵称相同，那就是你自己之前发过的弹幕；不要把它作为风格参考。",
    ]
    if fusion.scene_mode == "singer":
        lines.append("6) 当前是唱歌场景，可以参考弹幕发送“好听”或“打call”，严禁根据主播歌词回复。")
    lines.append("")
    lines.append("ordered_context: ")
    lines.append(context_json)
    return "\n".join(lines)

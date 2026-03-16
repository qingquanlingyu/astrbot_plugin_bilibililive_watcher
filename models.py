from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class DanmakuItem:
    uid: str
    nickname: str
    text: str
    ts: float
    timeline: str
    dedup_key: str
    event_type: str = "danmu"
    source: str = "history"


@dataclass(slots=True)
class ASRSegment:
    text: str
    ts_start: float
    ts_end: float
    conf: float
    wall_ts_start: float = 0.0
    wall_ts_end: float = 0.0


@dataclass(slots=True)
class FusionSummary:
    window_seconds: int
    danmaku_count: int
    asr_sentence_count: int = 0
    top_keywords: list[str] = field(default_factory=list)
    emotion: str = "neutral"
    asr_recent_topics: list[str] = field(default_factory=list)
    asr_samples: list[str] = field(default_factory=list)
    asr_confidence: float = 0.0
    scene_mode: str = "chat"
    constraints: list[str] = field(default_factory=list)
    singer_score: float = 0.0
    ordered_context: list[dict] = field(default_factory=list)

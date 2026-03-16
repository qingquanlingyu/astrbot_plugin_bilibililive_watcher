from __future__ import annotations

import re
from collections import Counter

try:  # pragma: no cover
    from .models import ASRSegment, DanmakuItem, FusionSummary
except ImportError:  # pragma: no cover
    from models import ASRSegment, DanmakuItem, FusionSummary

SINGER_KEYWORDS = (
    "好听",
    "打call",
    "妮莉安",
    "天籁之音",
    "/\\" # 打call弹幕常用
)

POSITIVE_WORDS = ("好", "棒", "喜欢", "稳", "强", "牛", "神", "爽", "赞", "可爱", "起飞")
NEGATIVE_WORDS = ("难听", "无聊", "尬", "拉胯", "吵", "烦", "烂")


class FusionEngine:
    def build_summary(
        self,
        danmaku_items: list[DanmakuItem],
        asr_segments: list[ASRSegment],
        window_seconds: int,
        singer_mode_enabled: bool,
        singer_mode_threshold: float,
    ) -> FusionSummary:
        danmaku_texts = [x.text.strip() for x in danmaku_items if x.text.strip()]
        asr_texts = [x.text.strip() for x in asr_segments if x.text.strip()]

        top_keywords = self._extract_top_keywords(danmaku_texts, limit=5)
        emotion = self._detect_emotion(danmaku_texts)
        asr_topics = self._extract_top_keywords(asr_texts, limit=3)
        asr_conf = self._avg_conf(asr_segments)
        singer_score = self._calc_singer_score(danmaku_texts, asr_texts)

        scene_mode = "chat"
        constraints: list[str] = []
        if singer_mode_enabled and singer_score >= max(0.0, min(1.0, singer_mode_threshold)):
            scene_mode = "singer"
            constraints = ["no_lyric_copy", "short_reaction"]

        return FusionSummary(
            window_seconds=window_seconds,
            danmaku_count=len(danmaku_items),
            asr_sentence_count=len(asr_segments),
            top_keywords=top_keywords,
            emotion=emotion,
            asr_recent_topics=asr_topics,
            asr_samples=asr_texts[-3:],
            asr_confidence=asr_conf,
            scene_mode=scene_mode,
            constraints=constraints,
            singer_score=singer_score,
        )

    def _calc_singer_score(self, danmaku_texts: list[str], asr_texts: list[str]) -> float:
        if not danmaku_texts and not asr_texts:
            return 0.0

        keyword_hits = 0
        for text in danmaku_texts:
            lower = text.lower()
            if any(k in lower for k in SINGER_KEYWORDS):
                keyword_hits += 1
        keyword_density = keyword_hits / max(1, len(danmaku_texts))

        short_lines = [x for x in asr_texts if len(x) <= 12]
        short_rate = len(short_lines) / max(1, len(asr_texts))

        rep_rate = self._repetition_rate(asr_texts)
        score = 0.55 * keyword_density + 0.2 * short_rate + 0.25 * rep_rate
        return max(0.0, min(1.0, score))

    def _repetition_rate(self, texts: list[str]) -> float:
        cleaned = [self._normalize_for_overlap(x) for x in texts if x.strip()]
        cleaned = [x for x in cleaned if x]
        if len(cleaned) < 2:
            return 0.0
        total_pairs = 0
        repeated_pairs = 0
        for i in range(len(cleaned) - 1):
            total_pairs += 1
            if cleaned[i] == cleaned[i + 1]:
                repeated_pairs += 1
        return repeated_pairs / max(1, total_pairs)

    def _extract_top_keywords(self, texts: list[str], limit: int) -> list[str]:
        if not texts:
            return []
        counter: Counter[str] = Counter()
        for text in texts:
            normalized = text.strip().lower()
            if not normalized:
                continue
            for kw in SINGER_KEYWORDS:
                if kw in normalized:
                    counter[kw] += 1
            for token in self._tokenize(normalized):
                if len(token) < 2:
                    continue
                if token.isdigit():
                    continue
                counter[token] += 1
        return [x for x, _ in counter.most_common(limit)]

    def _detect_emotion(self, texts: list[str]) -> str:
        if not texts:
            return "neutral"
        pos = 0
        neg = 0
        for text in texts:
            for w in POSITIVE_WORDS:
                if w in text:
                    pos += 1
            for w in NEGATIVE_WORDS:
                if w in text:
                    neg += 1
        if pos >= neg * 2 and pos > 0:
            return "positive"
        if neg > pos and neg > 0:
            return "negative"
        return "neutral"

    def _avg_conf(self, asr_segments: list[ASRSegment]) -> float:
        vals = [x.conf for x in asr_segments if x.conf > 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _normalize_for_overlap(self, text: str) -> str:
        text = re.sub(r"\s+", "", text.strip().lower())
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    def _tokenize(self, text: str) -> list[str]:
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{2,}", text)
        tokens: list[str] = []
        for chunk in chunks:
            if re.match(r"^[\u4e00-\u9fff]{2,}$", chunk):
                if len(chunk) <= 4:
                    tokens.append(chunk)
                else:
                    # 对较长中文片段做简单切分，避免整句作为“关键词”
                    for i in range(0, len(chunk) - 1):
                        tokens.append(chunk[i : i + 2])
            else:
                tokens.append(chunk)
        return tokens

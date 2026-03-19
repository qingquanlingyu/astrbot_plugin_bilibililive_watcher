from __future__ import annotations

import re
from collections import Counter

try:  # pragma: no cover
    from .models import ASRSegment, DanmakuItem, FusionSummary
except ImportError:  # pragma: no cover
    from models import ASRSegment, DanmakuItem, FusionSummary

DEFAULT_SINGER_KEYWORDS = (
    "好听",
    "打call",
    "天籁之音",
    "/\\",  # 打call弹幕常用
)


class FusionEngine:
    def build_summary(
        self,
        danmaku_items: list[DanmakuItem],
        asr_segments: list[ASRSegment],
        window_seconds: int,
        singer_mode_enabled: bool,
        singer_mode_keywords: list[str] | tuple[str, ...],
        singer_mode_window_seconds: int,
    ) -> FusionSummary:
        danmaku_texts = [x.text.strip() for x in danmaku_items if x.text.strip()]
        asr_texts = [x.text.strip() for x in asr_segments if x.text.strip()]
        singer_keywords = self._normalize_singer_keywords(singer_mode_keywords)
        singer_hit_keywords = self._collect_singer_hit_keywords(
            danmaku_items=danmaku_items,
            keywords=singer_keywords,
            window_seconds=singer_mode_window_seconds,
        )

        top_keywords = self._extract_top_keywords(danmaku_texts, limit=5, singer_keywords=singer_keywords)
        asr_topics = self._extract_top_keywords(asr_texts, limit=3)
        asr_conf = self._avg_conf(asr_segments)

        scene_mode = "chat"
        constraints: list[str] = []
        if singer_mode_enabled and singer_hit_keywords:
            scene_mode = "singer"
            constraints = ["no_lyric_copy", "short_reaction"]

        return FusionSummary(
            window_seconds=window_seconds,
            danmaku_count=len(danmaku_items),
            asr_sentence_count=len(asr_segments),
            top_keywords=top_keywords,
            asr_recent_topics=asr_topics,
            asr_samples=asr_texts[-3:],
            asr_confidence=asr_conf,
            scene_mode=scene_mode,
            constraints=constraints,
            singer_hit_keywords=singer_hit_keywords,
            singer_window_seconds=max(0, int(singer_mode_window_seconds or 0)),
        )

    def _normalize_singer_keywords(self, keywords: list[str] | tuple[str, ...]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        raw_items = list(keywords or DEFAULT_SINGER_KEYWORDS)
        for item in raw_items:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized or list(DEFAULT_SINGER_KEYWORDS)

    def _collect_singer_hit_keywords(
        self,
        *,
        danmaku_items: list[DanmakuItem],
        keywords: list[str],
        window_seconds: int,
    ) -> list[str]:
        if not danmaku_items or not keywords:
            return []
        latest_ts = max(float(item.ts or 0.0) for item in danmaku_items)
        cutoff_ts = latest_ts - max(0, int(window_seconds or 0))
        hits: list[str] = []
        seen: set[str] = set()
        for item in danmaku_items:
            if float(item.ts or 0.0) < cutoff_ts:
                continue
            text = str(item.text or "").strip().lower()
            if not text:
                continue
            for keyword in keywords:
                if keyword in text and keyword not in seen:
                    seen.add(keyword)
                    hits.append(keyword)
        return hits

    def _extract_top_keywords(
        self,
        texts: list[str],
        limit: int,
        singer_keywords: list[str] | None = None,
    ) -> list[str]:
        if not texts:
            return []
        counter: Counter[str] = Counter()
        keywords = list(singer_keywords or [])
        for text in texts:
            normalized = text.strip().lower()
            if not normalized:
                continue
            for kw in keywords:
                if kw in normalized:
                    counter[kw] += 1
            for token in self._tokenize(normalized):
                if len(token) < 2:
                    continue
                if token.isdigit():
                    continue
                counter[token] += 1
        return [x for x, _ in counter.most_common(limit)]

    def _avg_conf(self, asr_segments: list[ASRSegment]) -> float:
        vals = [x.conf for x in asr_segments if x.conf > 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

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

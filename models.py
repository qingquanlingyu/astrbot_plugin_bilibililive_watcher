from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass(slots=True)
class WatchConfig:
    enabled: bool
    debug: bool
    room_id: int
    auto_reply_enabled: bool
    reply_interval_seconds: int
    context_window_seconds: int
    danmaku_trigger_threshold: int
    asr_trigger_threshold: int
    target_umo: str
    target_platform_id: str
    target_type: str
    target_id: str
    generation_provider_id: str
    generation_persona_id: str
    generation_prompt_template: str
    max_reply_chars: int
    bilibili_cookie: str
    bilibili_cookie_source: str
    sync_to_bilibili_live: bool
    bili_login_cookie: str
    bili_login_uid: str
    bili_login_uname: str
    bili_login_refresh_token: str
    bili_login_saved_at: int
    pipeline_mode: str
    use_realtime_danmaku_ws: bool
    danmu_ws_auth_mode: str
    allow_buvid3_only: bool
    wbi_sign_enabled: bool
    audio_pull_protocol: str
    audio_pull_api_preference: str
    audio_http_headers_enabled: bool
    ffmpeg_path: str
    audio_sample_rate: int
    asr_backend: str
    asr_model_dir: str
    asr_vad_model_path: str
    asr_vad_threshold: float
    asr_vad_min_silence_duration: float
    asr_vad_min_speech_duration: float
    asr_vad_max_speech_duration: float
    asr_sense_voice_language: str
    asr_sense_voice_use_itn: bool
    asr_runtime_probe_required: bool
    asr_threads: int
    recording_enabled: bool = False
    recording_mode: str = "record_only"
    recording_segment_duration_seconds: int = 300
    recording_output_container: str = "mkv"
    recording_max_session_hours: int = 12
    storage_runtime_root: str = "/mnt/ssd/bilibili"
    clip_ai_enabled: bool = False
    clip_ai_prompt_template: str = ""
    publish_enabled: bool = False
    publish_default_visibility: str = "self_only"
    publish_max_retries: int = 3
    publish_retry_backoff_seconds: int = 300
    publish_default_tid: int = 0
    publish_default_tags: list[str] = field(default_factory=list)
    publish_use_tid_predict: bool = True
    publish_use_tag_recommendation: bool = True
    publish_cover_strategy: str = "midpoint_frame"
    publish_title_template: str = "{{room_title}} {{clip_range}} 切片"
    publish_desc_template: str = (
        "主播：{{anchor_name}}\n"
        "直播间：https://live.bilibili.com/{{real_room_id}}\n"
        "日期：{{clip_date}}\n\n"
        "{{auto_desc}}"
    )


@dataclass(slots=True)
class ChannelSendState:
    channel: str
    enabled: bool = False
    last_attempt_ts: float = 0.0
    last_success_ts: float = 0.0
    ok: bool | None = None
    summary: str = "never"
    error: str = ""
    text_preview: str = ""


@dataclass(slots=True)
class LoginRuntimeState:
    qrcode_key: str = ""
    url: str = ""
    image_path: str = ""
    status: str = "idle"
    message: str = ""
    started_ts: float = 0.0
    expires_at: float = 0.0
    completed_ts: float = 0.0


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
    asr_recent_topics: list[str] = field(default_factory=list)
    asr_samples: list[str] = field(default_factory=list)
    asr_confidence: float = 0.0
    scene_mode: str = "chat"
    constraints: list[str] = field(default_factory=list)
    singer_hit_keywords: list[str] = field(default_factory=list)
    singer_window_seconds: int = 0
    ordered_context: list[dict] = field(default_factory=list)

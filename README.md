# astrbot_plugin_bilibililive_watcher

定时监听 B 站直播间弹幕热度，并可融合直播语音 ASR；当触发条件满足时：

1. 把这段时间内的弹幕与语音识别语句按时间顺序注入模型上下文；
2. 使用当前会话的人设与模型生成一句简短弹幕风格文案；
3. 主动发送到指定群聊/私聊。

## 配置项

在插件配置中填写（可用 JSON/YAML 对应字段）：

```yaml
enabled: true
debug: false

# 直播间
room_id: 27484357

# 管线模式：仅弹幕 / 弹幕+ASR / 仅 ASR
pipeline_mode: "danmu_plus_asr"

# 触发节拍与上下文窗口
reply_interval_seconds: 15
context_window_seconds: 45
danmaku_trigger_threshold: 20
asr_trigger_threshold: 2

# 发送目标（二选一）
# 方式1：直接给完整 UMO（推荐）
# target_umo: "aiocqhttp:GroupMessage:123456789"

# 方式2：分字段拼接
target_platform_id: "default"
target_type: "group" # group / private
target_id: "123456789"

max_reply_chars: 60

# B站 cookie（防风控）
bili_cookie: ""
auto_load_cookie_from_file: true
bili_cookie_file: "~/.bilibili-cookie.json"

# 实时弹幕
use_realtime_danmaku_ws: true
danmu_ws_auth_mode: "signed_wbi"
allow_buvid3_only: true
wbi_sign_enabled: true

# 音频 / ASR
audio_enabled: true
audio_pull_protocol: "http_flv"
audio_pull_api_preference: "getRoomPlayInfo"
audio_http_headers_enabled: true
ffmpeg_path: "ffmpeg"
audio_sample_rate: 16000
asr_backend: "sherpa_onnx_rknn"
asr_strategy: "sensevoice_vad_offline" # 或 "streaming_zipformer" 回退
asr_model_dir: "./models/sherpa/rknn/sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
asr_vad_model_path: "./models/vad/silero_vad.onnx"
asr_vad_threshold: 0.3
asr_vad_min_silence_duration: 0.35
asr_vad_min_speech_duration: 0.25
asr_vad_max_speech_duration: 20.0
asr_sense_voice_language: "auto"
asr_sense_voice_use_itn: true
asr_runtime_probe_required: true
asr_threads: 1
asr_vad_enabled: false # 旧 streaming_zipformer 兼容字段
# 下列断句参数仅保留给旧 streaming_zipformer 回退链路
asr_sentence_pause_seconds: 0.8
asr_sentence_min_chars: 2

# 融合
singer_mode_enabled: true
singer_mode_threshold: 0.3
```

### 关键语义

- `reply_interval_seconds`：统一循环时间；每到这个时间点，插件检查一次是否满足触发条件
- `debug`：是否打印调试日志；开启后会输出实时弹幕、ASR 结果和最终 prompt
- `context_window_seconds`：保留最近多少秒的弹幕/ASR 历史作为上下文
- `danmaku_trigger_threshold`：累计弹幕条数阈值
- `asr_trigger_threshold`：累计 ASR 语句条数阈值
- `audio_pull_protocol`：音频流协议，可用 `http_flv` / `hls`
- `asr_strategy`：当前默认主链路为 `sensevoice_vad_offline`，可回退到 `streaming_zipformer`
- `asr_vad_model_path`：Silero VAD 模型文件，按插件目录相对路径解析
- `asr_vad_threshold` / `asr_vad_min_*` / `asr_vad_max_*`：控制 VAD 切段灵敏度、最短静音和最长单句
- `asr_sense_voice_language` / `asr_sense_voice_use_itn`：控制 SenseVoice 语言参数和 ITN
- `asr_sentence_pause_seconds`：ASR 停顿多久后，认为上一句结束
- `asr_sentence_min_chars`：避免过短碎句被单独算作一句
- `pipeline_mode=danmu_plus_asr` 时，需要**弹幕和 ASR 同时达标**才触发回复
- `pipeline_mode=asr_only` 时，只根据 ASR 阈值触发回复，不再要求弹幕条数达标
- 触发后只清空“待触发累计”，不会立刻清空上下文历史
- 若未显式配置 `context_window_seconds`，代码会兼容回退为 `reply_interval_seconds * 3`
- prompt 开头会给出主播昵称和直播间标题
- prompt 中会带 `ordered_context`，其中：
  - `source` 为中文 `"弹幕"` / `"主播"`
  - 按时间顺序排列，但不再把排序辅助时间字段直接暴露给模型
- 实时链路只记录真正的直播弹幕（`DANMU_MSG`）和 ASR 语句，不再把点赞/礼物/进房事件混入“弹幕上下文”
- 首次 history 轮询只用于建立去重基线，避免插件启动时把很早之前的旧弹幕误算进来
- 当 WS 已连上但暂时没有收到实时弹幕时，插件仍会继续轮询 history 作为兜底，避免“首轮后不再读弹幕”
- 新默认 ASR 路径是 `VAD 分段 + non-streaming SenseVoice`，更偏向长时间稳定的句段识别
- `streaming_zipformer` 旧链路仍保留，便于灰度切换和现场回退

## Cookie 说明

- 默认会尝试读取 `~/.bilibili-cookie.json`，格式兼容 `bilibili-danmu_sender`：
  `{"cookie": "...", "timestamp": 1234567890}`
- 也可以直接配置 `bili_cookie` 字符串（仍兼容旧字段 `bilibili_cookie`）。

## 指令

- `/biliwatch`：查看指令列表。
- `/biliwatch status`：查看当前插件运行状态和关键配置。
- `/biliwatch toggle [on|off]`：快速开关插件功能。
- `/biliwatch room <room_id>`：设置监听直播间号。
- `/biliwatch bind`：将发送目标绑定到当前会话（执行该指令的群/私聊）。

## 路由排查

- 如果日志出现 `cannot find platform for session ...`，说明会话前缀平台 ID 不匹配。
- 当前版本会自动从 AstrBot 已加载平台实例中发现可用 ID（常见是 `default`）并自动修正。
- 你也可以直接把 `target_umo` 设成当前聊天事件里的 `event.unified_msg_origin` 同格式值。

# astrbot_plugin_bilibililive_watcher

定时监听 B 站直播间弹幕与直播语音，融合上下文后触发模型生成一条简短回复。发布版支持两条发送通道：

1. 发送到 AstrBot 绑定会话。
2. 可选同步发送到 B 站直播弹幕。

默认升级行为保持不变：`sync_to_bilibili_live=false` 时，仍只发送到 AstrBot。

## 当前能力

- 稳定能力
  - 直播间弹幕监听与去重
  - 直播音频拉流 + ASR 融合
  - 定时触发生成短回复
  - 发送到 AstrBot 会话
  - 手工 `bili_cookie` 回退
  - RKNN 运行时探测与错误 wheel 诊断
- 实验性能力
  - B 站直播弹幕同步发送
  - 插件内二维码登录
- 配套SKILL
  - bilibili_live_context_fetcher，用于约束模型何时将直播上下文注入自身聊天会话

## 作者人工提示

1. 我是在RK3588上跑的，所以首先用的RKNN模型，不保证其他格式模型可以支持
2. 推荐下载`20-seconds-sense-voice`类模型，不推荐也不支持实时（曾经试过实时，但是准确性、断句都很差）
3. 使用`test_live_text_console.py [房间号] --asr`可以快速测试是否跑通

## 安装与前置

### AstrBot 自动安装的 Python 依赖

插件根目录已提供 `requirements.txt`。AstrBot 安装插件时会按官方机制安装：

- `aiohttp`
- `sherpa-onnx`

`requirements.txt` 已优先指向 sherpa-onnx 的 RKNN 发布来源，但最终是否拿到正确 RKNN wheel，仍以运行时探测结果为准。

### 仍需用户自行准备的外部前置

下列内容不会由插件自动安装：

- `ffmpeg`
  - 用于拉取直播音频。
- ASR 模型文件
  - 当前主支持 `.rknn` 模型。
  - 默认模型目录：`./models/sherpa/rknn/...`
  - 默认 VAD 模型：`./models/vad/silero_vad.onnx`
- 如果使用RKNN
  - 需要系统可提供 `librknnrt.so`。

### RKNN 主支持路线

发布版主支持：

- RK3588
- `sherpa-onnx` RKNN wheel
- `.rknn` 模型

CPU ONNX 仅作为回退路线，不作为首发主路线。

## 快速配置

```yaml
enabled: true
debug: false

room_id: 27484357
pipeline_mode: 2

# main_loop
reply_interval_seconds: 15
context_window_seconds: 45
danmaku_trigger_threshold: 20
asr_trigger_threshold: 2

# AstrBot 发送目标
# target_umo: "aiocqhttp:GroupMessage:123456789"
target_platform_id: "default"
target_type: "group"
target_id: "123456789"

# generation
provider_id: ""
persona_id: ""
prompt_template: "默认已内置标准直播弹幕生成模板"
max_reply_chars: 60

# B站 cookie 回退
bili_cookie: ""
bili_login_cookie: ""

# 新增：是否同步发送到 B 站直播弹幕
sync_to_bilibili_live: false

audio_pull_protocol: "http_flv"
ffmpeg_path: "ffmpeg"
audio_sample_rate: 16000

asr_backend: "sherpa_onnx_rknn"
asr_model_dir: "./models/sherpa/rknn/sherpa-onnx-rk3588-20-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"
asr_vad_model_path: "./models/vad/silero_vad.onnx"
asr_threads: 1

singer_mode_enabled: true
singer_mode_keywords: ["好听", "打call", "天籁之音", "/\\"]
singer_mode_window_seconds: 20
singer_mode_instruction: "6) 当前是唱歌场景，可以参考弹幕发送“好听”或“打call”，严禁根据主播歌词回复。"
```

以下行为已改为插件内部默认开启，不再建议手工配置：

- 实时弹幕优先使用 WebSocket，鉴权固定走 `signed_wbi`
- WebSocket 鉴权优先仅携带 `buvid3`，并启用 WBI 签名
- 音频拉流 API 固定优先 `getRoomPlayInfo`
- `ffmpeg` 拉流默认注入 `Referer/Origin/User-Agent/Cookie`
- SenseVoice 默认启用 ITN
- ASR 默认要求先通过运行时探测再启用
- 旧 `streaming_zipformer` 已移除，ASR 固定为 SenseVoice + VAD 离线链路

`pipeline_mode` 现改为整型配置：`0=danmu_only`、`1=asr_only`、`2=danmu_plus_asr`；非法值会回退为 `0`。

`singer_mode` 现改为规则型配置：只要最近 `singer_mode_window_seconds` 秒内的弹幕命中 `singer_mode_keywords` 任一关键词，就视为唱歌场景。

二维码登录成功后的 cookie / uid / uname / refresh_token 等字段仍会作为插件内部状态持久化。其中 `bili_login_cookie` 会继续暴露，便于你把它复制到 `bili_cookie` 做备份；如果 `bili_cookie` 原本为空，首次登录成功后也会自动写入同一份 cookie，避免插件重置后丢失登录态；其余登录内部字段不建议手工维护。

生成 prompt 时，插件会额外告诉模型当前自己在 B 站使用的昵称，用来识别 `ordered_context` 里哪些弹幕是自己之前发过的，避免连续复读。

`generation` 现在支持独立配置生成链路：

- `provider_id`
  - 单独指定用于生成弹幕的模型提供商；为空时跟随当前会话。
- `persona_id`
  - 单独指定用于生成弹幕的人设；为空时跟随当前会话。
- `prompt_template`
  - 自定义 prompt 模板，默认就是插件当前内置模板；支持双层花括号变量。
  - 常用变量：`{{anchor_name}}`、`{{room_title}}`、`{{room_id}}`、`{{self_bili_nickname}}`、`{{reply_length_limit}}`、`{{context_json}}`、`{{scene_mode}}`、`{{singer_mode_instruction}}`

`singer` 配置现在额外支持：

- `singer_mode_instruction`
  - 仅在识别为唱歌场景时附加到 prompt 中。
  - 可以在 `prompt_template` 里通过 `{{singer_mode_instruction}}` 引用；留空则不附加。

## 指令

- `/biliwatch help`
- `/biliwatch status`
- `/biliwatch toggle [on|off]`
- `/biliwatch room <room_id>`
- `/biliwatch bind`
- `/biliwatch sync-live [on|off]`
- `/biliwatch login`
- `/biliwatch login status`
- `/biliwatch logout`

### 二维码登录说明

- `/biliwatch login` 会发起官方 Web 二维码登录轮询。
- 插件会优先尝试直接发送二维码图片；如果当前环境缺少二维码依赖或平台不支持图片发送，会回退到登录 URL。
- 该能力标记为实验性；若 B 站链路不稳定，请直接回退到 `bili_cookie`。
- `/biliwatch logout` 只清除插件内保存的登录态；如果你另外配置了 `bili_cookie`，回退凭据仍可能继续生效。

## 状态输出

`/biliwatch status` 现在会额外展示：

- `bili_cookie_source`
  - `config` / `login` / `none`
- `bili_live_sync_enabled`
- `bili_login_status`
  - 是否已登录、账号脱敏信息、来源与失败摘要
- `bili_login_runtime`
  - 当前二维码登录轮询状态
- `astrbot_send`
- `bili_live_send`
  - 最近一次发送是否成功、失败摘要、最近时间与消息预览

## 关键语义

- `sync_to_bilibili_live=false`
  - 保持旧行为，只发 AstrBot。
- `sync_to_bilibili_live=true`
  - 同一次触发中，会同时尝试 AstrBot 和 B 站直播弹幕两个通道。
  - 任一通道失败不会阻断另一通道。
- Cookie 优先级
  - `bili_cookie`
  - `bili_login_cookie`

## 故障排查

### 1. `asr` 状态显示 wheel 不含 RKNN 支持

常见表现：

- `disabled(strategy=..., reason=current sherpa_onnx wheel has no RKNN support...)`

说明：

- 当前自动安装到的 `sherpa-onnx` 不是 RKNN wheel。
- 这不是“插件已安装好但 ASR 自然不可用”的正常状态，需要更换为正确的 RKNN wheel。

### 2. B 站同步发送失败

先看 `/biliwatch status` 中的：

- `bili_cookie_source`
- `bili_login_status`
- `bili_live_send`

常见原因：

- 未登录
- cookie 失效
- cookie 中缺少 `bili_jct`
- B 站接口风控或返回错误码

这类失败不会阻断 AstrBot 侧发送。

### 3. 二维码登录失败或超时

二维码登录是实验性能力。若出现：

- 一直停留在 `waiting_scan` / `waiting_confirm`
- `expired`
- `error`

请直接回退到：

- `bili_cookie`

### 4. 音频不可用

检查：

- `ffmpeg` 是否可执行
- 直播间是否正在开播
- 模型目录是否存在
- `silero_vad.onnx` 是否存在
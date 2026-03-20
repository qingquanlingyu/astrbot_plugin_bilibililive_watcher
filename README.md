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
  - 会话中读取直播上下文
  - RKNN 运行时探测与错误 wheel 诊断
- 实验性能力
  - B 站直播弹幕同步发送
  - 插件内二维码登录
- 配套SKILL
  - bilibili_live_context_fetcher，用于约束模型何时将直播上下文注入自身聊天会话

## 作者人工提示

- <font size="4">**本插件必须下载语音转文本模型**</font>
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

- <font size="4">**本插件必须自备 ffmpeg **</font>

- 推荐下载`20-seconds-sense-voice`类模型，不推荐也不支持实时（曾经试过实时，但是准确性、断句都很差）

- 使用`test_live_text_console.py [房间号] --asr`可以快速测试是否跑通

## 指令

下面这组命令基本覆盖了日常使用。推荐顺序通常是：先绑定目标会话，再设置直播间，然后按需微调轮询和触发阈值。

- `/biliwatch help`
  显示完整命令帮助，适合忘记命令时快速查看。
- `/biliwatch status`
  查看当前插件状态，包括直播间、发送目标、ASR 状态、WS 状态、最近触发时间、登录状态等。
- `/biliwatch toggle [on|off]`
  开关插件整体运行。
  不传参数时会在开启和关闭之间切换。
- `/biliwatch room <room_id>`
  设置监听的 B 站直播间号。
  可以填短号，也可以填真实房间号。
- `/biliwatch bind`
  把当前会话绑定为 AstrBot 侧的消息发送目标。
  一般建议先进入目标群聊或私聊后执行一次。
- `/biliwatch sync-live [on|off]`
  控制是否把生成结果同时发送到 B 站直播弹幕。
  默认建议关闭，先在 AstrBot 会话里观察效果。
- `/biliwatch reply-interval <seconds>`
  设置主循环间隔。
  这个值同时会影响常规 history 轮询频率，越小越及时，请求也越频繁。
- `/biliwatch context-window <seconds>`
  设置上下文保留窗口。
  值越大，模型能参考的近期弹幕和 ASR 越多，但也更容易混入旧话题。
- `/biliwatch danmaku-threshold <count>`
  设置触发生成前至少需要积累多少条弹幕。
  设为 `0` 表示不要求弹幕条数门槛。
- `/biliwatch asr-threshold <count>`
  设置触发生成前至少需要积累多少条 ASR 语句。
  只在启用了 ASR 的模式下生效；设为 `0` 表示不要求 ASR 条数门槛。
- `/biliwatch login`
  发起 B 站二维码登录。
  成功后插件会保存登录态，供直播弹幕发送等能力使用。
- `/biliwatch login status`
  查看二维码登录流程当前进度，以及当前账号登录状态。
- `/biliwatch logout`
  清除插件内保存的二维码登录态。
  如果你还手工配置了 `bili_cookie`，该回退凭据仍然可能继续生效。

### 常见使用流程

1. 在希望接收生成结果的群聊或私聊里执行 `/biliwatch bind`
2. 执行 `/biliwatch room <room_id>`
3. 用 `/biliwatch status` 确认配置是否生效
4. 如需调节灵敏度，再用：
   `reply-interval`、`context-window`、`danmaku-threshold`、`asr-threshold`
5. 如果想把内容真正发回 B 站直播间，再执行 `/biliwatch login` 和 `/biliwatch sync-live on`

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
  - 默认模型目录：`./models/sherpa/...`
  - 默认 VAD 模型：`./models/vad/silero_vad.onnx`
- 如果使用RKNN
  - 需要系统可提供 `librknnrt.so`。


### 二维码登录说明

- `/biliwatch login` 会发起官方 Web 二维码登录轮询。
- 插件会优先尝试直接发送二维码图片；如果当前环境缺少二维码依赖或平台不支持图片发送，会回退到登录 URL。
- 该能力标记为实验性；若 B 站链路不稳定，请直接回退到 `bili_cookie`。
- `/biliwatch logout` 只清除插件内保存的登录态；如果你另外配置了 `bili_cookie`，回退凭据仍可能继续生效。

## 默认配置

```yaml
enabled: true
debug: false

room_id: 0
pipeline_mode: 2

# main_loop
reply_interval_seconds: 30
context_window_seconds: 300
danmaku_trigger_threshold: 10
asr_trigger_threshold: 10

# AstrBot 发送目标
# target_umo: "aiocqhttp:GroupMessage:123456789"
target_platform_id: "default"
target_type: "group"
target_id: ""

# generation
provider_id: ""
persona_id: ""
prompt_template: "默认已内置标准直播弹幕生成模板"
max_reply_chars: 30

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

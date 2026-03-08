# astrbot_plugin_bilibililive_watcher

定时监听 B 站直播间弹幕热度，当指定时间窗口内弹幕数量超过阈值时：

1. 把这段时间的弹幕注入到模型上下文；
2. 使用当前会话的人设与模型生成一句简短弹幕风格文案；
3. 主动发送到指定群聊/私聊。

另外，提示词内已注入对 `B站弹幕发送 / bili_danmu_sender` 能力的要求：
如果当前模型具备该 skill，可在内部计划里同时向直播间发同款弹幕（不强制）。

## 配置项

在插件配置中填写（可用 JSON/YAML 对应字段）：

```yaml
enabled: true

# 直播间
room_id: 27484357

# 时间窗口（任选其一，window_seconds 优先）
window_seconds: 60
# window_minutes: 1

# 轮询周期（秒）
poll_interval_seconds: 15

# 触发阈值：窗口内弹幕数 >= 该值才触发模型
trigger_threshold: 20

# 触发冷却（秒）
trigger_cooldown_seconds: 60

# 发送目标（二选一）
# 方式1：直接给完整 UMO（推荐）
# target_umo: "aiocqhttp:GroupMessage:123456789"

# 方式2：分字段拼接
target_platform_id: "default"
target_type: "group" # group / private
target_id: "123456789"

# 模型输入与输出控制
max_context_danmaku: 40
max_history_messages: 12
max_reply_chars: 60
inject_bili_send_hint: true

# B站 cookie（防风控）
bilibili_cookie: ""
auto_load_cookie_from_file: true
bilibili_cookie_file: "~/.bilibili-cookie.json"
```

## Cookie 说明

- 默认会尝试读取 `~/.bilibili-cookie.json`，格式兼容 `bilibili-danmu_sender`：
  `{"cookie": "...", "timestamp": 1234567890}`
- 也可以直接配置 `bilibili_cookie` 字符串。

## 额外开关

- `inject_bili_send_hint`:
  - `true`：在提示词中注入“若具备 `bili_danmu_sender` 能力可同步发直播间”的要求。
  - `false`：不注入这条要求。

## 指令

- `/biliwatch_status`：查看当前插件运行状态和关键配置。
- `/biliwatch_set_room <room_id>`：设置监听直播间号。
- `/biliwatch_bind_here`：将发送目标绑定到当前会话（执行该指令的群/私聊）。

## 路由排查

- 如果日志出现 `cannot find platform for session ...`，说明会话前缀平台 ID 不匹配。
- 当前版本会自动从 AstrBot 已加载平台实例中发现可用 ID（常见是 `default`）并自动修正。
- 你也可以直接把 `target_umo` 设成当前聊天事件里的 `event.unified_msg_origin` 同格式值。

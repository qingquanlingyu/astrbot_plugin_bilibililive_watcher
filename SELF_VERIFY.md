# 需求

目前插件的b站弹幕流WebSocket链接存在问题，希望使用`test_live_text_console.py`做快速测试进行修复

# 自验证说明

当前建议优先通过 `test_live_text_console.py` 做长期自验证，不要直接依赖 AstrBot 主流程判断 ws 是否正常。

## 目的

这个脚本可以独立验证三件事：

- ws 是否真正连上
- ws 连上后是否持续收到实时弹幕
- `gethistory` 和 ws 的数量、时间是否大致对齐

## 基本命令

先确认环境里没有 `python` 别名问题，统一使用 `python3`。

只验证 ws：

```bash
python3 -u test_live_text_console.py 27484357 --no-history --ws-status-interval 5
```

同时对比 ws 和 `gethistory`：

```bash
python3 -u test_live_text_console.py 27484357 \
  --compare-history \
  --poll-interval 5 \
  --compare-interval 5 \
  --compare-window-seconds 20 \
  --ws-status-interval 5
```

如果需要带登录态验证：

```bash
python3 -u test_live_text_console.py 27484357 \
  --cookie-file ~/.bilibili-cookie.json \
  --compare-history \
  --poll-interval 5 \
  --compare-interval 5 \
  --compare-window-seconds 20 \
  --ws-status-interval 5
```

## 关键输出

`WS-STATUS` 用来看连接本身：

- `running=True connected=True`：ws 任务在跑，且已完成鉴权
- `msg_count`：当前进程内累计收到的 ws 弹幕数
- `last_msg_age`：距离最近一条 ws 弹幕过去多久
- `fatal`：ws 致命错误；正常时应为 `-`

`COMPARE` 用来看 ws 和 `gethistory` 是否对齐：

- `ws=当前窗口内ws事件数/去重后条数`
- `hist=当前窗口内history事件数/去重后条数`
- `overlap`：两路都出现过的消息数
- `ws_only`：只在 ws 看见
- `hist_only`：只在 history 看见
- `hist_timeline_last`：最近一条 history 弹幕自带的时间

## 建议观察方式

建议至少跑 5 到 10 分钟，不要只看启动后的前十几秒。

启动初期 `history` 往往会先出现一批 backlog，这不代表 ws 异常。判断重点是：

- ws 是否长期保持 `connected=True`
- `msg_count` 是否会持续增长
- `last_msg_age` 是否经常很长但 `history` 仍在更新
- `COMPARE` 是否长期只有 `hist_only`、几乎没有 `overlap`

## 当前经验结论

如果出现下面这种现象：

- `WS-STATUS` 一直显示 `connected=True`
- 但 `msg_count=0` 或增长极慢
- 同时 `gethistory` 还在持续拿到新弹幕

那么问题更可能在 ws 事件流本身，而不是 HTTP、房间号或展示逻辑。

## 交接给下一个 Codex

下一个 Codex 可以直接从下面这条命令开始长期观察：

```bash
python3 -u test_live_text_console.py 27484357 \
  --compare-history \
  --poll-interval 5 \
  --compare-interval 5 \
  --compare-window-seconds 20 \
  --ws-status-interval 5
```

如果要继续定位，优先检查：

- `bili_ws.py` 是否真的收到了 `op=5`
- 收到的原始 payload 里有哪些 `cmd`
- 是否存在 `DANMU_MSG` 之外的弹幕事件格式被漏解析

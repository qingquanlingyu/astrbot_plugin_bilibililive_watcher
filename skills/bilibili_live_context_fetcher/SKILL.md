---
name: bilibili_live_context_fetcher
description: 当你的上下文中存在<bili_live_room_state>块，且内部状态为"直播中"；并且用户询问了当前主播主播内容相关问题时调用。
---

# Bilibili Live Context Fetcher

## 前置条件

先确认这些条件成立：
- 已安装并配置 astrbot_plugin_bilibililive_watcher 连接插件
- 上下文中存在`<bili_live_room_state>`块，且内部状态为"直播中"

## 典型可调用场景
- “主播刚才说了什么”
- “现在直播间在聊什么”
- “刚刚弹幕都在刷什么”

## 调用方法

使用工具`bili_live_context_window`获取该直播间近期主播说的话和弹幕

单轮最多调用一次。拿到结果后，直接基于返回的窗口内容回答，不要重复调用

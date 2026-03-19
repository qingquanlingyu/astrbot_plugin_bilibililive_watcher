from __future__ import annotations

import importlib
import sys
import types
import unittest


def _install_fake_astrbot_modules():
    if "astrbot.api" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    core = types.ModuleType("astrbot.core")
    provider = types.ModuleType("astrbot.core.provider")
    entities = types.ModuleType("astrbot.core.provider.entities")

    class _Logger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    class _Filter:
        @staticmethod
        def command(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def llm_tool(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def on_llm_request(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class _MessageChain:
        def message(self, text):
            return self

    class _Star:
        def __init__(self, context=None):
            self.context = context

    def _register(*args, **kwargs):
        def decorator(cls):
            return cls

        return decorator

    class _Context:
        pass

    class _AstrMessageEvent:
        pass

    class _ProviderRequest:
        pass

    api.AstrBotConfig = dict
    api.logger = _Logger()
    event.AstrMessageEvent = _AstrMessageEvent
    event.MessageChain = _MessageChain
    event.filter = _Filter
    star.Context = _Context
    star.Star = _Star
    star.register = _register
    entities.ProviderRequest = _ProviderRequest

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.star"] = star
    sys.modules["astrbot.core"] = core
    sys.modules["astrbot.core.provider"] = provider
    sys.modules["astrbot.core.provider.entities"] = entities


_install_fake_astrbot_modules()
main = importlib.import_module("main")


class _FakeConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved = False

    def save_config(self):
        self.saved = True


class BiliWatcherConfigTests(unittest.TestCase):
    def _make_plugin(self, raw_config: dict) -> object:
        plugin = main.BilibiliLiveWatcherPlugin.__new__(main.BilibiliLiveWatcherPlugin)
        plugin.config = _FakeConfig(raw_config)
        return plugin

    def test_hidden_advanced_options_are_forced_to_internal_defaults(self):
        plugin = self._make_plugin(
            {
                "use_realtime_danmaku_ws": False,
                "danmu_ws_auth_mode": "history_only",
                "allow_buvid3_only": False,
                "wbi_sign_enabled": False,
                "audio_pull_api_preference": "playUrl",
                "audio_http_headers_enabled": False,
                "asr_sense_voice_use_itn": False,
                "asr_runtime_probe_required": False,
            }
        )

        cfg = plugin._load_config()

        self.assertTrue(cfg.use_realtime_danmaku_ws)
        self.assertEqual(cfg.danmu_ws_auth_mode, "signed_wbi")
        self.assertTrue(cfg.allow_buvid3_only)
        self.assertTrue(cfg.wbi_sign_enabled)
        self.assertEqual(cfg.audio_pull_api_preference, "getRoomPlayInfo")
        self.assertTrue(cfg.audio_http_headers_enabled)
        self.assertTrue(cfg.asr_sense_voice_use_itn)
        self.assertTrue(cfg.asr_runtime_probe_required)

    def test_pipeline_mode_accepts_numeric_values_and_invalid_falls_back_to_danmu_only(self):
        plugin = self._make_plugin({"pipeline_mode": 2})
        self.assertEqual(plugin._load_config().pipeline_mode, "danmu_plus_asr")

        plugin = self._make_plugin({"pipeline_mode": 1})
        self.assertEqual(plugin._load_config().pipeline_mode, "asr_only")

        plugin = self._make_plugin({"pipeline_mode": 0})
        self.assertEqual(plugin._load_config().pipeline_mode, "danmu_only")

        plugin = self._make_plugin({"pipeline_mode": 999})
        self.assertEqual(plugin._load_config().pipeline_mode, "danmu_only")

    def test_singer_mode_keywords_and_window_are_loaded(self):
        plugin = self._make_plugin(
            {
                "singer_mode_keywords": ["好听", "打call", "好听"],
                "singer_mode_window_seconds": 30,
            }
        )

        cfg = plugin._load_config()

        self.assertEqual(cfg.singer_mode_keywords, ["好听", "打call"])
        self.assertEqual(cfg.singer_mode_window_seconds, 30)

    def test_danmaku_trigger_threshold_allows_zero(self):
        plugin = self._make_plugin({"pipeline_mode": 0, "danmaku_trigger_threshold": 0})

        cfg = plugin._load_config()

        self.assertEqual(cfg.danmaku_trigger_threshold, 0)
        plugin._buffer = []
        plugin._asr_buffer = []
        self.assertEqual(plugin._get_trigger_blockers(cfg), [])

    def test_cookie_file_fallback_is_ignored(self):
        plugin = self._make_plugin(
            {
                "bili_cookie": "",
                "bili_cookie_file": "/tmp/legacy-cookie.json",
                "auto_load_cookie_from_file": True,
                "bili_login_cookie": "login-cookie",
            }
        )

        cfg = plugin._load_config()

        self.assertEqual(cfg.bilibili_cookie, "login-cookie")
        self.assertEqual(cfg.bilibili_cookie_source, "login")

    def test_save_config_prunes_hidden_legacy_keys(self):
        plugin = self._make_plugin(
            {
                "room_id": 123,
                "bili_cookie_file": "/tmp/legacy-cookie.json",
                "bilibili_cookie_file": "/tmp/legacy-cookie-compat.json",
                "auto_load_cookie_from_file": True,
                "audio_enabled": False,
                "singer_mode_threshold": 0.9,
                "use_realtime_danmaku_ws": False,
                "audio_pull_api_preference": "playUrl",
                "asr_runtime_probe_required": False,
                "bili_login_cookie": "persist-me",
            }
        )

        saved = plugin._save_config_if_possible()

        self.assertTrue(saved)
        self.assertTrue(plugin.config.saved)
        self.assertEqual(plugin.config.get("room_id"), 123)
        self.assertEqual(plugin.config.get("bili_login_cookie"), "persist-me")
        self.assertNotIn("bili_cookie_file", plugin.config)
        self.assertNotIn("bilibili_cookie_file", plugin.config)
        self.assertNotIn("auto_load_cookie_from_file", plugin.config)
        self.assertNotIn("audio_enabled", plugin.config)
        self.assertNotIn("singer_mode_threshold", plugin.config)
        self.assertNotIn("use_realtime_danmaku_ws", plugin.config)
        self.assertNotIn("audio_pull_api_preference", plugin.config)
        self.assertNotIn("asr_runtime_probe_required", plugin.config)


if __name__ == "__main__":
    unittest.main()

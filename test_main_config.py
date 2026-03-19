from __future__ import annotations

import asyncio
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
prompting = importlib.import_module("prompting")


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

    def test_nested_config_groups_are_loaded(self):
        plugin = self._make_plugin(
            {
                "global": {
                    "enabled": False,
                    "debug": True,
                    "room_id": 5566,
                    "pipeline_mode": 1,
                    "sync_to_bilibili_live": True,
                },
                "main_loop": {
                    "reply_interval_seconds": 12,
                    "context_window_seconds": 48,
                    "danmaku_trigger_threshold": 0,
                    "asr_trigger_threshold": 3,
                },
                "sender": {
                    "target_umo": "aiocqhttp:GroupMessage:123",
                    "target_platform_id": "onebot",
                    "target_type": "group",
                    "target_id": "123",
                },
                "generation": {
                    "provider_id": "provider.test",
                    "persona_id": "persona.live",
                    "prompt_template": "主播={{anchor_name}} 限制={{reply_length_limit}}",
                    "max_reply_chars": 42,
                },
                "user_auth": {
                    "bili_cookie": "manual-cookie",
                    "bili_login_cookie": "login-cookie",
                    "bili_login_uid": "1001",
                    "bili_login_uname": "测试账号",
                    "bili_login_refresh_token": "refresh-1",
                    "bili_login_saved_at": 123456,
                },
                "asr": {
                    "audio_pull_protocol": "hls",
                    "ffmpeg_path": "/usr/bin/ffmpeg",
                    "audio_sample_rate": 22050,
                    "asr_backend": "sherpa_onnx_cpu_onnx",
                    "asr_model_dir": "./models/custom",
                    "asr_vad_model_path": "./models/custom_vad.onnx",
                    "asr_vad_threshold": 0.45,
                    "asr_vad_min_silence_duration": 0.5,
                    "asr_vad_min_speech_duration": 0.3,
                    "asr_vad_max_speech_duration": 12.0,
                    "asr_sense_voice_language": "zh",
                    "asr_threads": -1,
                },
                "singer": {
                    "singer_mode_enabled": False,
                    "singer_mode_keywords": ["合唱", "再来一首"],
                    "singer_mode_window_seconds": 15,
                    "singer_mode_instruction": "唱歌时只发打call类短句，别接歌词",
                },
            }
        )

        cfg = plugin._load_config()

        self.assertFalse(cfg.enabled)
        self.assertTrue(cfg.debug)
        self.assertEqual(cfg.room_id, 5566)
        self.assertEqual(cfg.pipeline_mode, "asr_only")
        self.assertTrue(cfg.sync_to_bilibili_live)
        self.assertEqual(cfg.reply_interval_seconds, 12)
        self.assertEqual(cfg.context_window_seconds, 48)
        self.assertEqual(cfg.danmaku_trigger_threshold, 0)
        self.assertEqual(cfg.asr_trigger_threshold, 3)
        self.assertEqual(cfg.target_umo, "aiocqhttp:GroupMessage:123")
        self.assertEqual(cfg.target_platform_id, "onebot")
        self.assertEqual(cfg.target_type, "group")
        self.assertEqual(cfg.target_id, "123")
        self.assertEqual(cfg.generation_provider_id, "provider.test")
        self.assertEqual(cfg.generation_persona_id, "persona.live")
        self.assertEqual(cfg.generation_prompt_template, "主播={{anchor_name}} 限制={{reply_length_limit}}")
        self.assertEqual(cfg.max_reply_chars, 42)
        self.assertEqual(cfg.bilibili_cookie, "manual-cookie")
        self.assertEqual(cfg.bilibili_cookie_source, "config")
        self.assertEqual(cfg.bili_login_cookie, "login-cookie")
        self.assertEqual(cfg.bili_login_uid, "1001")
        self.assertEqual(cfg.bili_login_uname, "测试账号")
        self.assertEqual(cfg.bili_login_refresh_token, "refresh-1")
        self.assertEqual(cfg.bili_login_saved_at, 123456)
        self.assertEqual(cfg.audio_pull_protocol, "hls")
        self.assertEqual(cfg.ffmpeg_path, "/usr/bin/ffmpeg")
        self.assertEqual(cfg.audio_sample_rate, 22050)
        self.assertEqual(cfg.asr_backend, "sherpa_onnx_cpu_onnx")
        self.assertTrue(cfg.asr_model_dir.endswith("/models/custom"))
        self.assertTrue(cfg.asr_vad_model_path.endswith("/models/custom_vad.onnx"))
        self.assertEqual(cfg.asr_vad_threshold, 0.45)
        self.assertEqual(cfg.asr_vad_min_silence_duration, 0.5)
        self.assertEqual(cfg.asr_vad_min_speech_duration, 0.3)
        self.assertEqual(cfg.asr_vad_max_speech_duration, 12.0)
        self.assertEqual(cfg.asr_sense_voice_language, "zh")
        self.assertEqual(cfg.asr_threads, -1)
        self.assertFalse(cfg.singer_mode_enabled)
        self.assertEqual(cfg.singer_mode_keywords, ["合唱", "再来一首"])
        self.assertEqual(cfg.singer_mode_window_seconds, 15)
        self.assertEqual(cfg.singer_mode_instruction, "唱歌时只发打call类短句，别接歌词")

    def test_generation_prompt_template_defaults_to_builtin_template(self):
        plugin = self._make_plugin({})

        cfg = plugin._load_config()

        self.assertEqual(cfg.generation_prompt_template, prompting.DEFAULT_FUSED_PROMPT_TEMPLATE)

    def test_singer_group_falls_back_to_legacy_other_group(self):
        plugin = self._make_plugin(
            {
                "other": {
                    "singer_mode_enabled": False,
                    "singer_mode_keywords": ["好稳"],
                    "singer_mode_window_seconds": 9,
                    "singer_mode_instruction": "只夸唱功",
                }
            }
        )

        cfg = plugin._load_config()

        self.assertFalse(cfg.singer_mode_enabled)
        self.assertEqual(cfg.singer_mode_keywords, ["好稳"])
        self.assertEqual(cfg.singer_mode_window_seconds, 9)
        self.assertEqual(cfg.singer_mode_instruction, "只夸唱功")

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

    def test_persist_login_success_syncs_backup_cookie_when_manual_cookie_is_empty(self):
        plugin = self._make_plugin({"user_auth": {"bili_cookie": ""}})
        plugin._http = None

        asyncio.run(
            plugin._persist_login_success(
                cookie="SESSDATA=abc",
                refresh_token="refresh-token",
                account=main.BiliLoginAccount(
                    is_logged_in=True,
                    uid="12345",
                    uname="测试用户",
                    source="login",
                ),
            )
        )

        self.assertEqual(plugin.config["user_auth"]["bili_cookie"], "SESSDATA=abc")
        self.assertEqual(plugin.config["user_auth"]["bili_login_cookie"], "SESSDATA=abc")
        self.assertEqual(plugin.config["user_auth"]["bili_login_refresh_token"], "refresh-token")
        self.assertEqual(plugin.config["user_auth"]["bili_login_uid"], "12345")
        self.assertEqual(plugin.config["user_auth"]["bili_login_uname"], "测试用户")
        self.assertGreater(plugin.config["user_auth"]["bili_login_saved_at"], 0)
        self.assertTrue(plugin.config.saved)

    def test_persist_login_success_keeps_existing_manual_cookie(self):
        plugin = self._make_plugin({"user_auth": {"bili_cookie": "manual-cookie"}})
        plugin._http = None

        asyncio.run(
            plugin._persist_login_success(
                cookie="SESSDATA=abc",
                refresh_token="refresh-token",
                account=main.BiliLoginAccount(
                    is_logged_in=True,
                    uid="12345",
                    uname="测试用户",
                    source="login",
                ),
            )
        )

        self.assertEqual(plugin.config["user_auth"]["bili_cookie"], "manual-cookie")
        self.assertEqual(plugin.config["user_auth"]["bili_login_cookie"], "SESSDATA=abc")

    def test_save_config_prunes_hidden_legacy_keys(self):
        plugin = self._make_plugin(
            {
                "room_id": 123,
                "bili_cookie_file": "/tmp/legacy-cookie.json",
                "bilibili_cookie_file": "/tmp/legacy-cookie-compat.json",
                "auto_load_cookie_from_file": True,
                "audio_enabled": False,
                "asr_strategy": "streaming_zipformer",
                "asr_vad_enabled": True,
                "asr_sentence_pause_seconds": 1.2,
                "asr_sentence_min_chars": 5,
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
        self.assertNotIn("asr_strategy", plugin.config)
        self.assertNotIn("asr_vad_enabled", plugin.config)
        self.assertNotIn("asr_sentence_pause_seconds", plugin.config)
        self.assertNotIn("asr_sentence_min_chars", plugin.config)
        self.assertNotIn("singer_mode_threshold", plugin.config)
        self.assertNotIn("use_realtime_danmaku_ws", plugin.config)
        self.assertNotIn("audio_pull_api_preference", plugin.config)
        self.assertNotIn("asr_runtime_probe_required", plugin.config)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from bili_http import BiliHttpClient


class _FakeHeaders:
    def __init__(self, values: list[str]):
        self._values = list(values)

    def getall(self, key: str, default=None):
        if key.lower() == "set-cookie":
            return list(self._values)
        return list(default or [])


class BiliHttpClientHelperTests(unittest.TestCase):
    def setUp(self):
        self.client = BiliHttpClient(session=None)  # type: ignore[arg-type]

    def test_normalize_qr_poll_status(self):
        self.assertEqual(
            self.client._normalize_qr_poll_status(raw_code=86101, raw_message="未扫码"),
            "waiting_scan",
        )
        self.assertEqual(
            self.client._normalize_qr_poll_status(raw_code=86090, raw_message="已扫码未确认"),
            "waiting_confirm",
        )
        self.assertEqual(
            self.client._normalize_qr_poll_status(raw_code=86038, raw_message="二维码已失效"),
            "expired",
        )
        self.assertEqual(
            self.client._normalize_qr_poll_status(raw_code=0, raw_message="0"),
            "confirmed",
        )

    def test_extract_cookie_pairs_from_url_supports_nested_go_url(self):
        raw_url = (
            "https://passport.bilibili.com/crossDomain?"
            "go_url=https%3A%2F%2Fexample.com%2Fcallback%3FSESSDATA%3Dsess"
            "%26bili_jct%3Dcsrf%26DedeUserID%3D12345"
        )

        pairs = self.client._extract_cookie_pairs_from_url(raw_url)

        self.assertEqual(
            pairs,
            {
                "SESSDATA": "sess",
                "bili_jct": "csrf",
                "DedeUserID": "12345",
            },
        )

    def test_extract_cookie_from_qr_poll_payload_merges_multiple_sources(self):
        payload = {
            "cookie_info": {
                "cookies": [
                    {"name": "SESSDATA", "value": "sess"},
                    {"name": "DedeUserID", "value": "12345"},
                ]
            },
            "url": "https://example.com/callback?bili_jct=csrf&sid=abc",
        }
        headers = _FakeHeaders(["buvid3=buvid; Path=/; HttpOnly"])

        cookie = self.client._extract_cookie_from_qr_poll_payload(
            data=payload,
            response_headers=headers,
        )

        self.assertEqual(
            cookie,
            "SESSDATA=sess; bili_jct=csrf; DedeUserID=12345; sid=abc; buvid3=buvid",
        )

    def test_extract_cookie_value(self):
        cookie = "SESSDATA=sess; bili_jct=csrf; DedeUserID=12345"
        self.assertEqual(self.client._extract_cookie_value(cookie, "bili_jct"), "csrf")
        self.assertEqual(self.client._extract_cookie_value(cookie, "missing"), "")


if __name__ == "__main__":
    unittest.main()

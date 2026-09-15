import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import CookieJar
from yarl import URL

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import (
    CSRFTokenNotFoundError,
    ShuiyuanModel,
    _apply_cookies,
)


class ForumRequestChainTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cancelled_waiter_does_not_wedge_later_requests(self):
        import asyncio

        calls = []

        class FakeSession:
            def __init__(self):
                self.headers = {}

            async def get(self, *args, **kwargs):
                calls.append(args[0])
                await asyncio.sleep(0.05)
                return SimpleNamespace(release=Mock())

        with (
            patch.object(ShuiyuanModel, "_shared_session", FakeSession()),
            patch.object(ShuiyuanModel, "_request_chain", None),
            patch.object(ShuiyuanModel, "_last_request_ts", 0.0),
            patch.object(ShuiyuanModel, "_request_interval", 0.01),
        ):
            first = asyncio.create_task(
                ShuiyuanModel._rate_limited_request("get", "https://example.test/1")
            )
            await asyncio.sleep(0.01)  # the first request is in flight
            second = asyncio.create_task(
                ShuiyuanModel._rate_limited_request("get", "https://example.test/2")
            )
            await asyncio.sleep(0.01)  # the second one waits behind it
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            await first
            third = await asyncio.wait_for(
                ShuiyuanModel._rate_limited_request("get", "https://example.test/3"), 1
            )
        self.assertIsNotNone(third)
        self.assertEqual(calls, ["https://example.test/1", "https://example.test/3"])


class ForumAuthTests(unittest.IsolatedAsyncioTestCase):
    async def check_response(self, url, status, html):
        response = SimpleNamespace(
            url=URL(url),
            status=status,
            text=AsyncMock(return_value=html),
            release=Mock(),
        )
        session = SimpleNamespace(headers={})
        with (
            patch.object(ShuiyuanModel, "_shared_session", session),
            patch.object(
                ShuiyuanModel, "_rate_limited_request", AsyncMock(return_value=response)
            ),
        ):
            try:
                await ShuiyuanModel._update_cookies()
            finally:
                response.release.assert_called_once()
        return session.headers

    async def test_meta_attribute_order_and_quotes(self):
        for html in (
            '<meta name="csrf-token" content="token">',
            "<meta content='token' data-extra='yes' name='csrf-token' />",
        ):
            headers = await self.check_response(
                "https://shuiyuan.sjtu.edu.cn/", 200, html
            )
            self.assertEqual(headers["X-CSRF-Token"], "token")

    async def test_login_redirect_does_not_leak_url_or_accept_foreign_token(self):
        with self.assertRaises(CSRFTokenNotFoundError) as error:
            await self.check_response(
                "https://jaccount.sjtu.edu.cn/login?secret=do-not-log",
                200,
                '<meta name="csrf-token" content="foreign">',
            )
        self.assertIn("redirected", str(error.exception))
        self.assertNotIn("do-not-log", str(error.exception))
        self.assertNotIn("foreign", str(error.exception))

    async def test_http_failure_and_missing_token(self):
        for status, message in ((403, "HTTP 403"), (200, "no CSRF token")):
            with self.assertRaisesRegex(CSRFTokenNotFoundError, message):
                await self.check_response(
                    "https://shuiyuan.sjtu.edu.cn/", status, "no token"
                )


class TimeRangeSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_range_search_delegates_to_the_retrying_helper(self):
        # The public helper used to call a removed _retry_wrapper, so every
        # search_posts_by_time call failed with AttributeError.
        model = ShuiyuanModel.__new__(ShuiyuanModel)
        calls = []

        async def fake_search(topic_id, after_date=None, before_date=None):
            calls.append((topic_id, after_date, before_date))
            return {"标题": []}

        model._search_post_details_by_time_range_and_topic = fake_search
        result = await model.search_post_details_by_time_range_and_topic(
            7, "2026-01-01", "2026-01-02"
        )
        self.assertEqual(result, {"标题": []})
        self.assertEqual(calls, [(7, "2026-01-01", "2026-01-02")])


class CookieJarDistributionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cookies_reach_forum_and_jaccount_during_sso(self):
        # get_cookies.ipynb writes a flat jAccount cookie dict; the SSO bounce
        # through jaccount.sjtu.edu.cn requires these cookies at both hosts.
        jar = CookieJar()
        _apply_cookies(jar, {"JSESSIONID": "s", "JAAuthCookie": "c"})
        forum_sent = set(jar.filter_cookies(URL("https://shuiyuan.sjtu.edu.cn")))
        sso_sent = set(jar.filter_cookies(URL("https://jaccount.sjtu.edu.cn")))
        self.assertIn("JSESSIONID", forum_sent)
        self.assertIn("JSESSIONID", sso_sent)
        self.assertIn("JAAuthCookie", sso_sent)

    async def test_cookies_stay_off_unrelated_hosts(self):
        jar = CookieJar()
        _apply_cookies(jar, {"JSESSIONID": "s"})
        sent = set(jar.filter_cookies(URL("https://example.com")))
        self.assertEqual(sent, set())

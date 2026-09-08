import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from yarl import URL

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import (
    CSRFTokenNotFoundError,
    ShuiyuanModel,
)


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

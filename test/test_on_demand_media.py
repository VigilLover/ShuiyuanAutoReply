import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import ToolMessage

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.bootstrap.settings import DeepSeekApiFormat
from shuiyuan_auto_reply.domain.tool_error import ReadFailure
from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    MentionDeepSeekModel,
)
from shuiyuan_auto_reply.features.mention.mention_multimodal import (
    normalize_shuiyuan_image_url,
)
from shuiyuan_auto_reply.infrastructure.image_transport import (
    ImageDownloadError,
    cached_media_attempt,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel


class OnDemandMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_retrieval_and_reply_do_not_prepare_images(self):
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        model.api_format = DeepSeekApiFormat.CHAT_COMPLETIONS
        model.vision_media = SimpleNamespace(
            prepare_tool_output=AsyncMock(), prepare_forum_url=AsyncMock()
        )
        state = {
            "image_inputs": [],
            "reply_to_post_number": 7,
            "messages": [
                ToolMessage(
                    name="get_post",
                    content="![image](upload://x.png)",
                    tool_call_id="1",
                )
            ],
        }
        self.assertEqual(
            await model._collect_tool_output_images(state), {"image_inputs": []}
        )
        self.assertEqual(
            await model._load_replied_post_images(state), {"image_inputs": []}
        )
        model.vision_media.prepare_tool_output.assert_not_called()
        model.vision_media.prepare_forum_url.assert_not_called()

    async def test_missing_floor_is_not_retried(self):
        model = ShuiyuanModel.__new__(ShuiyuanModel)
        model._rate_limited_request = AsyncMock(
            return_value=SimpleNamespace(
                status=200, json=AsyncMock(return_value={"post_stream": {"posts": []}})
            )
        )
        with self.assertRaises(ReadFailure) as failure:
            await model.get_post_details_by_post_number(42, 19)
        self.assertFalse(failure.exception.retryable)
        model._rate_limited_request.assert_awaited_once()

    async def test_terminal_media_failure_shared_between_paths(self):
        token = current_turn.set(TurnResults())
        a, b = AsyncMock(side_effect=ImageDownloadError(404)), AsyncMock()

        @cached_media_attempt
        async def reference(url):
            return await a(url)

        @cached_media_attempt
        async def vision(url):
            return await b(url)

        try:
            with self.assertRaises(ImageDownloadError):
                await reference("https://example.org/image.png")
            with self.assertRaises(ImageDownloadError):
                await vision("https://example.org/image.png")
            b.assert_not_called()
        finally:
            current_turn.reset(token)

    def test_secure_original_stays_in_authenticated_forum_path(self):
        url = "https://shuiyuan.sjtu.edu.cn/secure-uploads/original/4X/a/photo.png"
        self.assertEqual(normalize_shuiyuan_image_url(url), url)
        self.assertIsNone(normalize_shuiyuan_image_url("https://example.org/photo.png"))

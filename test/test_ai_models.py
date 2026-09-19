"""
Connectivity and wiring checks for the DeepSeek chat model.

Usage:
    # Live connectivity (requires DEEPSEEK_API_KEY in .env):
    python -m pytest test/test_ai_models.py -v --run-live
"""

import logging
import os
import unittest
from unittest.mock import MagicMock, patch

import dotenv
import pytest
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_DS_BASE = "https://api.deepseek.com"


@pytest.mark.live
class TestDeepSeekDirectAPI(unittest.IsolatedAsyncioTestCase):
    """Test direct connectivity to the DeepSeek API."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

    async def test_configured_model(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")
        client = AsyncOpenAI(api_key=api_key, base_url=_DS_BASE)
        try:
            response = await client.chat.completions.create(
                model=ProviderSettings().deepseek_model,
                messages=[
                    {"role": "user", "content": "请回复'连通正常'。只输出这三个字。"}
                ],
            )
            text = response.choices[0].message.content
            self.assertTrue(text and text.strip())
        finally:
            await client.close()


class TestMentionDeepSeekVisionModel(unittest.IsolatedAsyncioTestCase):
    """The mention model binds exactly one DeepSeek chat model."""

    def setUp(self):
        self._patches = []

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _mock_agent(self):
        p = patch(
            "shuiyuan_auto_reply.features.mention.mention_chat_model.MentionChatModel.initialize_agent",
            new_callable=MagicMock,
        )
        p.start()
        self._patches.append(p)

    async def test_deepseek_model_uses_single_vision_llm(self):
        from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
            MentionDeepSeekModel,
        )
        from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

        self._mock_agent()
        settings = ProviderSettings(deepseek_api_key="test-key")
        with patch.object(ShuiyuanModel, "__init__", lambda self: None):
            model = MentionDeepSeekModel(MagicMock(), provider_settings=settings)

        self.assertIsInstance(model.llm, ChatOpenAI)
        self.assertEqual(model.llm.model_name, settings.deepseek_model)
        self.assertTrue(model.supports_multimodal)


if __name__ == "__main__":
    unittest.main(verbosity=2)

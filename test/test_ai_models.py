"""
Comprehensive tests for AI models: DeepSeek, Tongyi, and FallbackLLM mechanism.

Usage:
    # Run all tests (requires DEEPSEEK_API_KEY and DASHSCOPE_API_KEY in .env):
    python -m pytest test/test_ai_models.py -v

    # Run specific test class:
    python -m pytest test/test_ai_models.py::TestDeepSeekDirectAPI -v

Coverage:
    - Direct API connectivity for DeepSeek primary/fallback models (from env vars)
    - Direct API connectivity for Tongyi primary/fallback models (from env vars)
    - FallbackLLM: primary-LLM-failure → secondary-LLM-takes-over
    - MentionDeepSeekModel internal fallback wired correctly
    - MentionTongyiModel internal fallback wired correctly
"""

import os
import sys
import unittest
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

import dotenv
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi
from shuiyuan_auto_reply.features.mention.mention_chat_model import FallbackLLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Defaults (mirror those in the model modules) ────────────────────────

_DS_DEFAULT = "deepseek-v4-pro"
_DS_FALLBACK = "deepseek-v4-flash"
_DS_BASE = "https://api.deepseek.com"

_TY_DEFAULT = "qwen3.5-plus-2026-02-15"
_TY_FALLBACK = "qwen3.5-plus"
_TY_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _ds_model(key: str, default: str) -> str:
    return os.getenv(key, default)


def _ty_model(key: str, default: str) -> str:
    return os.getenv(key, default)


# ── Direct API Tests ─────────────────────────────────────────────────────

@pytest.mark.live
class TestDeepSeekDirectAPI(unittest.IsolatedAsyncioTestCase):
    """Test direct connectivity to DeepSeek API."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

    async def _do_chat(self, model: str, label: str):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key, base_url=_DS_BASE)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "请回复'连通正常'。只输出这三个字。"}],
            )
            text = response.choices[0].message.content
            logging.info("[DeepSeek %s] model=%s response=%s", label, model, text)
            self.assertIsNotNone(text)
            self.assertTrue(text.strip())
        finally:
            await client.close()

    async def test_primary_model(self):
        model = _ds_model("DEEPSEEK_MENTION_MODEL", _DS_DEFAULT)
        await self._do_chat(model, "primary")

    async def test_fallback_model(self):
        model = _ds_model("DEEPSEEK_MENTION_FALLBACK_MODEL", _DS_FALLBACK)
        await self._do_chat(model, "fallback")


@pytest.mark.live
class TestTongyiDirectAPI(unittest.IsolatedAsyncioTestCase):
    """Test direct connectivity to Tongyi (DashScope) API."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

    async def _do_chat(self, model: str, label: str):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            self.skipTest("DASHSCOPE_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key, base_url=_TY_BASE)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "请回复'连通正常'。只输出这三个字。"}],
            )
            text = response.choices[0].message.content
            logging.info("[Tongyi %s] model=%s response=%s", label, model, text)
            self.assertIsNotNone(text)
            self.assertTrue(text.strip())
        finally:
            await client.close()

    async def test_primary_model(self):
        model = _ty_model("DASHSCOPE_MENTION_MODEL", _TY_DEFAULT)
        await self._do_chat(model, "primary")

    async def test_fallback_model(self):
        model = _ty_model("DASHSCOPE_MENTION_FALLBACK_MODEL", _TY_FALLBACK)
        await self._do_chat(model, "fallback")


# ── FallbackLLM Tests ────────────────────────────────────────────────────

class FailingLLM:
    """A mock LLM that always raises an exception (simulates a broken primary)."""

    def __init__(self, exc_type=Exception, msg="mock primary failure"):
        self._exc_type = exc_type
        self._msg = msg

    async def ainvoke(self, *args, **kwargs):
        raise self._exc_type(self._msg)

    def invoke(self, *args, **kwargs):
        raise self._exc_type(self._msg)

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.live
class TestFallbackLLM(unittest.IsolatedAsyncioTestCase):
    """Test the FallbackLLM mechanism in isolation."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

    async def test_fallback_triggers_when_primary_fails(self):
        """When primary raises, fallback should take over."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")

        fallback_model = _ds_model("DEEPSEEK_MENTION_FALLBACK_MODEL", _DS_FALLBACK)
        failing_primary = FailingLLM(msg="Simulated primary failure")
        real_fallback = ChatOpenAI(
            model=fallback_model,
            api_key=api_key,
            base_url=_DS_BASE,
            max_retries=1,
        )
        fallback_llm = FallbackLLM(failing_primary, real_fallback)

        result = await fallback_llm.ainvoke("回复'fallback成功'。只输出这三个字，不要多说。")
        text = result.content if hasattr(result, "content") else str(result)
        logging.info("[FallbackLLM] result: %s", text)
        self.assertIsNotNone(text)
        self.assertTrue(text.strip())

    async def test_bind_tools_returns_fallbackllm(self):
        """bind_tools should preserve the fallback wrapping."""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")

        primary_model = _ds_model("DEEPSEEK_MENTION_MODEL", _DS_DEFAULT)
        fallback_model = _ds_model("DEEPSEEK_MENTION_FALLBACK_MODEL", _DS_FALLBACK)
        primary = ChatOpenAI(model=primary_model, api_key=api_key, base_url=_DS_BASE)
        fallback = ChatOpenAI(model=fallback_model, api_key=api_key, base_url=_DS_BASE)
        llm = FallbackLLM(primary, fallback)

        bound = llm.bind_tools([])
        from langchain_core.runnables import RunnableLambda
        self.assertIsInstance(bound, RunnableLambda)


# ── MentionDeepSeekModel Fallback Test ───────────────────────────────────

class TestMentionDeepSeekVisionModel(unittest.IsolatedAsyncioTestCase):
    """Test MentionDeepSeekModel internal fallback wiring."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

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
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            self.skipTest("DEEPSEEK_API_KEY not set")

        from shuiyuan_auto_reply.features.mention.mention_deepseek_model import MentionDeepSeekModel
        from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

        self._mock_agent()

        with patch.object(ShuiyuanModel, "__init__", lambda self: None):
            model = MentionDeepSeekModel(MagicMock())

        self.assertIsInstance(model.llm, ChatOpenAI)
        self.assertNotIsInstance(model.llm, FallbackLLM)
        self.assertEqual(model.llm.model_name, "deepseek-v4-flash-vision-exp")
        self.assertTrue(model.supports_multimodal)


# ── MentionTongyiModel Fallback Test ────────────────────────────────────

class TestMentionTongyiModelFallback(unittest.IsolatedAsyncioTestCase):
    """Test MentionTongyiModel internal fallback wiring."""

    @classmethod
    def setUpClass(cls):
        dotenv.load_dotenv()

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

    async def test_tongyi_model_has_fallback_llm(self):
        """MentionTongyiModel.llm should be a FallbackLLM with two ChatTongyi instances."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            self.skipTest("DASHSCOPE_API_KEY not set")

        from shuiyuan_auto_reply.features.mention.mention_tongyi_model import MentionTongyiModel
        from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

        self._mock_agent()

        with patch.object(ShuiyuanModel, "__init__", lambda self: None):
            model = MentionTongyiModel(MagicMock())

        self.assertIsInstance(model.llm, FallbackLLM)
        self.assertIsInstance(model.llm.primary, ChatOpenAI)
        self.assertIsInstance(model.llm.fallback, ChatOpenAI)

        expected_primary = _ty_model("DASHSCOPE_MENTION_MODEL", _TY_DEFAULT)
        expected_fallback = _ty_model("DASHSCOPE_MENTION_FALLBACK_MODEL", _TY_FALLBACK)
        self.assertEqual(model.llm.primary.model_name, expected_primary)
        self.assertEqual(model.llm.fallback.model_name, expected_fallback)
        logging.info(
            "[TongyiModel] primary=%s, fallback=%s",
            model.llm.primary.model_name,
            model.llm.fallback.model_name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

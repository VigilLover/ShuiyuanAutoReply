"""Opt-in-by-secret live contracts for DeepSeek's Responses endpoint."""

import base64
import os
import unittest
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
_MODEL = (
    os.getenv("DEEPSEEK_MENTION_MODEL", DEEPSEEK_DEFAULT_MODEL).strip()
    or DEEPSEEK_DEFAULT_MODEL
)
_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
).decode()


@unittest.skipUnless(_API_KEY, "DEEPSEEK_API_KEY is not configured")
class DeepSeekResponsesLiveContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncOpenAI(api_key=_API_KEY, base_url=DEEPSEEK_BASE_URL)

    async def asyncTearDown(self):
        await self.client.close()

    async def test_plain_text_and_all_reasoning_efforts(self):
        for effort in ("none", "high", "max"):
            with self.subTest(effort=effort):
                response = await self.client.responses.create(
                    model=_MODEL,
                    input="只回复 OK",
                    reasoning={"effort": effort},
                    max_output_tokens=128,
                )
                self.assertTrue(response.output_text.strip())

    async def test_user_image(self):
        response = await self.client.responses.create(
            model=_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "描述这张图片的主色"},
                        {"type": "input_image", "image_url": _PNG_DATA_URL},
                    ],
                }
            ],
            reasoning={"effort": "none"},
            max_output_tokens=256,
        )
        self.assertTrue(response.output_text.strip())

    async def test_forced_function_call_with_image_output(self):
        tools = [
            {
                "type": "function",
                "name": "inspect_fixture",
                "description": "读取测试图片",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        first = await self.client.responses.create(
            model=_MODEL,
            input="调用工具读取测试图片，然后说明图片内容。",
            tools=tools,
            tool_choice="required",
            # DeepSeek thinking mode rejects forced function tool_choice.
            reasoning={"effort": "none"},
            max_output_tokens=512,
        )
        function_call = next(
            item for item in first.output if item.type == "function_call"
        )
        replay = [item.model_dump(exclude_none=True) for item in first.output]
        replay.append(
            {
                "type": "function_call_output",
                "call_id": function_call.call_id,
                "output": [
                    {"type": "input_text", "text": "测试图片如下"},
                    {"type": "input_image", "image_url": _PNG_DATA_URL},
                ],
            }
        )
        second = await self.client.responses.create(
            model=_MODEL,
            input=replay,
            tools=tools,
            tool_choice="auto",
            reasoning={"effort": "none"},
            max_output_tokens=512,
        )
        self.assertTrue(second.output_text.strip())

    async def test_forced_official_web_search(self):
        response = await self.client.responses.create(
            model=_MODEL,
            input="搜索今天的公开科技新闻并用一句话回答。",
            tools=[{"type": "web_search"}],
            tool_choice="required",
            reasoning={"effort": "high"},
            max_output_tokens=1024,
        )
        self.assertTrue(any(item.type == "web_search_call" for item in response.output))
        self.assertTrue(response.output_text.strip())

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.bootstrap.settings import (
    DeepSeekApiFormat,
    ProviderSettings,
)
from shuiyuan_auto_reply.domain import VisualMediaArtifact
from shuiyuan_auto_reply.features.mention.deepseek_vision import DeepSeekVisionInput
from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    DEEPSEEK_DEFAULT_MODEL,
    DeepSeekChatOpenAI,
    MentionDeepSeekModel,
    _mk_deepseek_llm,
    build_deepseek_responses_content,
)


def _settings(
    api_format: DeepSeekApiFormat,
    *,
    thinking: str = "enabled",
    effort: str = "max",
    max_tokens: str | None = "2048",
) -> ProviderSettings:
    return ProviderSettings(
        deepseek_api_key="test-key",
        deepseek_api_format=api_format,
        deepseek_thinking=thinking,
        deepseek_reasoning_effort=effort,
        _deepseek_max_tokens=max_tokens,
    )


def _image(block: dict, source_url: str = "https://cdn.example/image.png"):
    artifact = VisualMediaArtifact(
        artifact_id="artifact-1",
        mime_type="image/png",
        local_path="/tmp/artifact-1.png",
        byte_count=10,
        source_kind="web_search",
        source_url=source_url,
    )
    return DeepSeekVisionInput(
        source_url=source_url,
        source_kind="web_search",
        content_block=block,
        artifact=artifact,
        description="测试图片",
    )


class DeepSeekRequestContractTests(unittest.TestCase):
    def test_chat_completions_payload_contract_is_preserved(self):
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(DeepSeekApiFormat.CHAT_COMPLETIONS),
        )
        content = [
            {"type": "text", "text": "看图"},
            {"type": "file", "file_id": "file_123"},
            {"type": "image_url", "image_url": {"url": "https://example/a.png"}},
        ]
        messages = [
            HumanMessage(content=content),
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "provider reasoning"},
                tool_calls=[
                    {"name": "lookup", "args": {"q": "x"}, "id": "call_1"}
                ],
            ),
        ]

        payload = model._get_request_payload(messages)

        self.assertIsInstance(model, DeepSeekChatOpenAI)
        self.assertEqual(payload["model"], DEEPSEEK_DEFAULT_MODEL)
        self.assertEqual(payload["messages"][0]["content"], content)
        self.assertEqual(
            payload["messages"][1]["reasoning_content"], "provider reasoning"
        )
        self.assertEqual(payload["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertEqual(payload["max_tokens"], 2048)

    def test_responses_maps_reasoning_and_max_output_tokens(self):
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(
                DeepSeekApiFormat.RESPONSES,
                thinking="enabled",
                effort="high",
                max_tokens="4096",
            ),
        )

        payload = model._get_request_payload([HumanMessage(content="hello")])

        self.assertNotIsInstance(model, DeepSeekChatOpenAI)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertIn("input", payload)
        self.assertNotIn("messages", payload)
        self.assertNotIn("previous_response_id", payload)
        self.assertNotIn("conversation", payload)
        self.assertNotIn("store", payload)

    def test_disabled_thinking_maps_to_none(self):
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(
                DeepSeekApiFormat.RESPONSES,
                thinking="disabled",
                max_tokens=None,
            ),
        )

        payload = model._get_request_payload([HumanMessage(content="hello")])

        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertNotIn("max_output_tokens", payload)

    def test_responses_supports_url_data_url_and_file_id_images(self):
        images = [
            _image(
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example/url.png"},
                },
                "https://cdn.example/url.png",
            ),
            _image(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                "data:image/png;base64,AAAA",
            ),
            _image(
                {"type": "file", "file_id": "file_vision_1"},
                "artifact://file-vision-1",
            ),
        ]
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(DeepSeekApiFormat.RESPONSES),
        )

        content = build_deepseek_responses_content("描述图片", images)
        payload = model._get_request_payload([HumanMessage(content=content)])
        input_images = [
            block
            for block in payload["input"][0]["content"]
            if block["type"] == "input_image"
        ]

        self.assertEqual(
            input_images,
            [
                {"type": "input_image", "image_url": "https://cdn.example/url.png"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                {"type": "input_image", "file_id": "file_vision_1"},
            ],
        )

    def test_reasoning_and_function_output_are_replayed_with_call_id(self):
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(DeepSeekApiFormat.RESPONSES),
        )
        reasoning = {"type": "reasoning", "id": "rs_1", "summary": []}
        messages = [
            HumanMessage(content="call a tool"),
            AIMessage(
                content=[reasoning],
                tool_calls=[{"name": "lookup", "args": {}, "id": "call_1"}],
            ),
            ToolMessage(
                content=[
                    {"type": "input_text", "text": "tool text"},
                    {"type": "input_image", "file_id": "file_tool_1"},
                ],
                tool_call_id="call_1",
                name="lookup",
            ),
        ]

        payload = model._get_request_payload(messages)

        self.assertIn(reasoning, payload["input"])
        function_call = next(
            item for item in payload["input"] if item["type"] == "function_call"
        )
        function_output = next(
            item
            for item in payload["input"]
            if item["type"] == "function_call_output"
        )
        self.assertEqual(function_call["call_id"], "call_1")
        self.assertEqual(function_output["call_id"], "call_1")
        self.assertEqual(
            function_output["output"][-1],
            {"type": "input_image", "file_id": "file_tool_1"},
        )

    def test_native_web_search_binding_is_provider_side(self):
        model = _mk_deepseek_llm(
            "test-key",
            DEEPSEEK_DEFAULT_MODEL,
            _settings(DeepSeekApiFormat.RESPONSES),
        )

        bound = model.bind_tools([{"type": "web_search"}], tool_choice="auto")

        self.assertEqual(bound.kwargs["tools"], [{"type": "web_search"}])
        self.assertEqual(bound.kwargs["tool_choice"], "auto")

    def test_final_text_excludes_reasoning_and_search_control_blocks(self):
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        raw = [
            {"type": "reasoning", "summary": [{"text": "hidden"}]},
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {"type": "output_text", "text": "公开回答"},
            {"type": "text", "text": "。"},
        ]

        self.assertEqual(model.parse_model_output(raw), "公开回答。")


class DeepSeekToolImageContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_web_search_is_hidden_and_never_added_to_tool_node(self):
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        model.provider_settings = SimpleNamespace(mcp_server_url=None)
        model.enabled_tools = None
        model.disabled_mcp_tools = set()
        model.state_store = SimpleNamespace(replace_tool_catalog=AsyncMock())
        model.prompt_scope = SimpleNamespace(value="web")
        model._load_shuiyuan_tools = MagicMock(return_value=[])
        model.memory_model = SimpleNamespace(initialize=AsyncMock(), tools=[])
        model.openai_tools = []
        model.hidden_provider_tools = [{"type": "web_search"}]
        model.provider_tool_choice = "auto"
        bound = SimpleNamespace(with_retry=MagicMock(return_value="bound"))
        model.llm = SimpleNamespace(bind_tools=MagicMock(return_value=bound))
        model._build_graph = MagicMock(return_value="graph")

        await model.initialize_agent()

        model.llm.bind_tools.assert_called_once_with(
            [{"type": "web_search"}], tool_choice="auto"
        )
        self.assertEqual(model.tools, [])
        model.state_store.replace_tool_catalog.assert_awaited_once_with("web", [])

    async def test_tool_image_replaces_matching_tool_message_not_user_message(self):
        image = _image({"type": "file", "file_id": "file_tool_1"})
        original = ToolMessage(
            content="tool text",
            tool_call_id="call_1",
            name="lookup",
            artifact={"source": "fixture"},
            status="success",
        )
        model = MentionDeepSeekModel.__new__(MentionDeepSeekModel)
        model.api_format = DeepSeekApiFormat.RESPONSES
        model.vision_media = SimpleNamespace(
            prepare_tool_output=AsyncMock(return_value=[image])
        )

        result = await model._collect_tool_output_images(
            {
                "image_inputs": [],
                "messages": [original],
                "conversation_id": "conversation-1",
            }
        )

        self.assertEqual(len(result["messages"]), 1)
        replacement = result["messages"][0]
        self.assertIsInstance(replacement, ToolMessage)
        self.assertEqual(replacement.id, original.id)
        self.assertEqual(replacement.tool_call_id, "call_1")
        self.assertEqual(replacement.name, "lookup")
        self.assertEqual(replacement.status, "success")
        self.assertEqual(replacement.artifact, {"source": "fixture"})
        self.assertEqual(
            replacement.content[-1],
            {"type": "input_image", "file_id": "file_tool_1"},
        )
        self.assertFalse(any(isinstance(item, HumanMessage) for item in result["messages"]))

"""Offline graph regression: generic forum lookup and user resolution, no image scenario branching."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.chat_pipeline import ChatOrchestrator
from shuiyuan_auto_reply.features.mention.mention_chat_model import (
    MentionChatModel,
    describe_model_failure,
)
from shuiyuan_auto_reply.shuiyuan.objects import User


class OfflineChat(MentionChatModel):
    def __init__(self, model):
        self.model = model
        self.username = "bot"
        self.supports_multimodal = False
        self.state_store = None
        self.prompt_scope = PromptScope.FORUM
        self._histories = {}
        self._history_access = {}
        self.memory_model = SimpleNamespace(
            enabled=False, memory_key=str, graph_config=lambda key: {}
        )
        self.pipeline = ChatOrchestrator(self)
        self.tools = self._load_shuiyuan_tools()
        self.graph = self._build_graph()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{recent_msgs}"),
                MessagesPlaceholder("chat_history"),
                MessagesPlaceholder("messages"),
            ]
        )
        self.prompts = []
        self.llm_with_tools = SimpleNamespace(ainvoke=self.respond)

    async def _retrieve_style_context(self, state):
        return {"context": ""}

    async def _load_long_term_memory(self, state):
        return {"long_term_memory": ""}

    async def respond(self, prompt):
        self.prompts.append(prompt.to_messages())
        n = len(self.prompts)
        if n == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "first",
                        "name": "users",
                        "args": {"username": "Alice", "include_avatar": True},
                    }
                ],
            )
        if n == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "again",
                        "name": "users",
                        "args": {"username": "Alice", "include_avatar": True},
                    }
                ],
            )
        return AIMessage(content="已读取指定帖子并确认用户资料。")

    def parse_model_output(self, raw_output):
        return str(raw_output or "")


class ForumAgentFlowTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def finalizer_state():
        return {
            "user": User(id=1, username="requester", name=None),
            "topic_id": None,
            "reply_to_post_number": None,
            "conversation": "请总结",
            "chat_history": [],
            "messages": [HumanMessage(content="请总结")],
            "target_post": None,
            "recent_msgs": "无近期回帖记录",
            "context": "",
            "long_term_memory": "无相关长期记忆",
            "supports_multimodal": False,
            "image_inputs": [],
        }

    def test_provider_failure_diagnostic_is_bounded_and_redacted(self):
        error = RuntimeError(
            "request failed with Authorization: Bearer token-value and sk-example123456"
        )
        error.response = SimpleNamespace(
            status_code=402,
            text='{"error":"balance exhausted","api_key":"body-secret"}',
        )

        detail = describe_model_failure(error)

        self.assertIn("HTTP 402", detail)
        self.assertIn("balance exhausted", detail)
        self.assertNotIn("token-value", detail)
        self.assertNotIn("sk-example123456", detail)
        self.assertNotIn("body-secret", detail)

    async def test_target_preload_and_repeated_queries_through_real_graph(self):
        post = SimpleNamespace(
            id=100,
            topic_id=42,
            post_number=7,
            reply_to_post_number=3,
            user_id=1,
            username="author",
            name=None,
            raw="名单：@Alice",
            cooked="<p>名单：@Alice</p>",
        )
        model = SimpleNamespace(
            get_post_details_by_post_number=AsyncMock(return_value=post),
            read_topic_post_page=AsyncMock(return_value=("Topic", [post], 1, False)),
            get_user_by_username=AsyncMock(
                return_value=SimpleNamespace(
                    id=9, username="Alice", name=None, avatar_template="/a/{size}.png"
                )
            ),
        )
        runtime = OfflineChat(model)
        result = await runtime.get_pumpkin_response(
            42,
            7,
            "读取被回复帖子中的用户资料",
            User(id=1, username="requester", name=None),
        )
        self.assertIn("确认用户资料", result)
        model.get_post_details_by_post_number.assert_awaited_once_with(42, 7)
        # The second, identical read is answered from this turn's cache; the replay must
        # still pair with its own call, because a reused message id used to collapse the
        # pair on the next model request.
        model.get_user_by_username.assert_awaited_once_with("Alice")
        pairs: dict[str, bool] = {}
        for message in runtime.prompts[2]:
            for tool_call in getattr(message, "tool_calls", None) or []:
                pairs[tool_call["id"]] = False
            answered = getattr(message, "tool_call_id", None)
            if answered:
                pairs[answered] = True
        self.assertTrue(pairs)
        self.assertEqual(
            [call_id for call_id, answered in pairs.items() if not answered],
            [],
        )
        first = "\n".join(str(m.content) for m in runtime.prompts[0])
        self.assertIn("名单：@Alice", first)
        self.assertIn('"reply_to": "forum:42/3"', first)
        self.assertIsNone(current_turn.get())

    async def test_new_tools_expose_parameters_and_managed_description(self):
        runtime = OfflineChat(SimpleNamespace())
        catalog = {tool.name: tool for tool in runtime.tools}
        properties = catalog["forum_read"].args_schema.model_json_schema()["properties"]
        self.assertIn("cursor", properties)
        # get_post absorbs the global-id lookup; search_user absorbs the id lookup.
        self.assertIn("post_id", properties)
        self.assertIn(
            "user_id",
            catalog["users"].args_schema.model_json_schema()["properties"],
        )
        self.assertIn(
            "references",
            catalog["generate_image"].args_schema.model_json_schema()["properties"],
        )
        self.assertNotIn(
            "prepare_image_references", catalog["generate_image"].description
        )
        # One entry point per capability: near-synonym tools made the model retry the
        # same read through a different name instead of using the result it had.
        for removed in (
            "get_post",
            "recent_posts",
            "search_posts",
            "search_user",
            "inspect_images",
            "read_tool_result",
        ):
            self.assertNotIn(removed, catalog)

    async def test_settings_tool_list_has_no_stale_names(self):
        from shuiyuan_auto_reply.interfaces.api.app import RUNTIME_TOOL_NAMES

        runtime = OfflineChat(SimpleNamespace())
        registered = {tool.name for tool in runtime.tools}
        # The settings page falls back to this list before the first agent run; a
        # removed tool left behind here would be advertised but never callable.
        # inspect_images needs a multimodal provider and the memory tools come from
        # the memory model, neither of which this harness registers.
        self.assertEqual(
            set(RUNTIME_TOOL_NAMES) - registered,
            {"search_mention_memory", "manage_mention_memory"},
        )

    async def test_generated_artifact_is_delivered_without_internal_notice(self):
        from shuiyuan_auto_reply.application.tool_results import TurnResults
        from shuiyuan_auto_reply.domain import GeneratedImageArtifact

        runtime = OfflineChat(SimpleNamespace())
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            artifact = GeneratedImageArtifact(
                "generated-id", "image/png", "/tmp/fake.png", 1
            )
            result = await runtime._finalize_response(
                {"messages": [AIMessage(content="")], "generated_artifacts": [artifact]}
            )
            self.assertIn("![生成图片](artifact://generated-id)", result["final_text"])
            self.assertNotIn("未纳入：Bob", result["final_text"])
        finally:
            current_turn.reset(token)

    async def test_tool_markup_is_not_accepted_as_final_text(self):
        runtime = OfflineChat(SimpleNamespace())
        result = await runtime._finalize_response(
            {
                "messages": [
                    AIMessage(content='<tool_call>{"name":"forum_search"}</tool_call>')
                ],
                "generated_artifacts": [],
            }
        )
        self.assertEqual(result["final_text"], "")

    async def test_finalizer_uses_clean_evidence_and_retry_context(self):
        runtime = OfflineChat(SimpleNamespace())
        invalid = (
            '<｜｜DSML｜｜ invoke name="forum_search">'
            '<｜｜DSML｜｜ parameter name="cursor">secret-cursor</｜｜DSML｜｜ parameter>'
        )
        runtime.llm = SimpleNamespace(
            ainvoke=AsyncMock(
                side_effect=[AIMessage(content=invalid), AIMessage(content="总结正文")]
            )
        )
        turn = TurnResults()
        turn.progress.phase = "final"
        turn.observe(
            {
                "items": [
                    {
                        "ref": "forum:42/1",
                        "author": "Alice",
                        "text": "已核实内容",
                    }
                ]
            },
            tool="forum_read",
        )
        token = current_turn.set(turn)
        state = {
            "user": User(id=1, username="requester", name=None),
            "topic_id": None,
            "reply_to_post_number": None,
            "conversation": "请总结话题",
            "chat_history": [HumanMessage(content="正常历史")],
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "old-call",
                            "name": "forum_search",
                            "args": {"query": "旧查询"},
                        }
                    ],
                ),
                ToolMessage(
                    content="旧工具结果",
                    tool_call_id="old-call",
                    name="forum_search",
                ),
            ],
            "target_post": None,
            "recent_msgs": "无近期回帖记录",
            "context": "",
            "long_term_memory": "无相关长期记忆",
            "supports_multimodal": False,
            "image_inputs": [],
        }
        try:
            first = await runtime._call_model(state)
            state["messages"] += first["messages"]
            self.assertEqual(
                (await runtime._finalize_response(state))["final_text"], ""
            )
            second = await runtime._call_model(state)
            state["messages"] += second["messages"]
            self.assertEqual(
                (await runtime._finalize_response(state))["final_text"], "总结正文"
            )
        finally:
            current_turn.reset(token)

        self.assertEqual(runtime.llm.ainvoke.await_count, 2)
        for call in runtime.llm.ainvoke.await_args_list:
            messages = call.args[0].to_messages()
            self.assertFalse(any(isinstance(item, ToolMessage) for item in messages))
            self.assertFalse(
                any(getattr(item, "tool_calls", None) for item in messages)
            )
            joined = "\n".join(str(item.content) for item in messages)
            self.assertIn("已核实内容", joined)
            self.assertIn("【可用上下文】", joined)
            self.assertIn("不要描述查询、调用、失败、重试或核实过程", joined)
            self.assertNotIn("【已核实资料】", joined)
            self.assertNotIn("secret-cursor", joined)

    async def test_investigation_prompt_keeps_tool_failures_internal(self):
        runtime = OfflineChat(SimpleNamespace())
        runtime.llm_with_tools = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content="直接回答"))
        )
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            await runtime._call_model(self.finalizer_state())
        finally:
            current_turn.reset(token)

        prompt = runtime.llm_with_tools.ainvoke.await_args.args[0].to_messages()
        joined = "\n".join(str(item.content) for item in prompt)
        self.assertIn("工具失败只用于调整内部策略", joined)
        self.assertIn("最终回答不得描述查询、调用、失败、重试或核实过程", joined)

    async def test_investigation_model_failure_gets_one_text_only_recovery(self):
        runtime = OfflineChat(SimpleNamespace())
        runtime.llm_with_tools = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("temporary provider failure"))
        )
        runtime.llm = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content="恢复后的正文"))
        )
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            result = await runtime._call_model(self.finalizer_state())
        finally:
            current_turn.reset(token)

        self.assertEqual(result["messages"][0].content, "恢复后的正文")
        self.assertEqual(turn.progress.phase, "final")
        runtime.llm_with_tools.ainvoke.assert_awaited_once()
        runtime.llm.ainvoke.assert_awaited_once()

    async def test_final_provider_failure_raises_without_completed_event(self):
        runtime = OfflineChat(SimpleNamespace())
        runtime.llm_with_tools = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("initial failure"))
        )
        runtime.llm = SimpleNamespace(
            ainvoke=AsyncMock(side_effect=RuntimeError("HTTP 402 balance exhausted"))
        )
        turn = TurnResults()
        token = current_turn.set(turn)
        events = AsyncMock()
        try:
            with patch(
                "shuiyuan_auto_reply.features.mention.mention_chat_model.emit_event",
                events,
            ):
                with self.assertRaisesRegex(RuntimeError, "balance exhausted"):
                    await runtime._call_model(self.finalizer_state())
        finally:
            current_turn.reset(token)

        event_names = [call.args[0] for call in events.await_args_list]
        self.assertEqual(event_names.count("model.failed"), 2)
        self.assertNotIn("model.completed", event_names)

    async def test_second_invalid_finalizer_result_fails_with_readable_error(self):
        runtime = OfflineChat(SimpleNamespace())
        invalid = '<tool_call>{"name":"forum_search"}</tool_call>'
        runtime.llm = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=AIMessage(
                    content=invalid,
                    usage_metadata={
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6,
                    },
                )
            )
        )

        async def graph_response(*_args, **_kwargs):
            current_turn.get().progress.phase = "final"
            return {
                "user": User(id=1, username="requester", name=None),
                "topic_id": None,
                "reply_to_post_number": None,
                "conversation": "总结",
                "chat_history": [],
                "messages": [AIMessage(content=invalid)],
                "target_post": None,
                "recent_msgs": "无近期回帖记录",
                "context": "",
                "long_term_memory": "无相关长期记忆",
                "supports_multimodal": False,
                "image_inputs": [],
                "generated_artifacts": [],
                "final_text": "",
                "raw_output": invalid,
            }

        runtime.graph = SimpleNamespace(ainvoke=graph_response)
        events = AsyncMock()
        with patch(
            "shuiyuan_auto_reply.features.mention.mention_chat_model.emit_event",
            events,
        ):
            with self.assertRaisesRegex(RuntimeError, "模型未生成有效正文"):
                await runtime.get_pumpkin_response(
                    None,
                    None,
                    "总结",
                    User(id=1, username="requester", name=None),
                    session_id="web-session",
                    load_forum_context=False,
                )
        event_names = [call.args[0] for call in events.await_args_list]
        self.assertIn("model.failed", event_names)
        self.assertNotIn("model.completed", event_names)
        self.assertIn("usage.recorded", event_names)

    def test_plain_dsml_reference_is_valid_text(self):
        self.assertFalse(OfflineChat._contains_tool_markup("这里讨论 DSML 协议。"))
        self.assertTrue(
            OfflineChat._contains_tool_markup(
                '<｜｜DSML｜｜ invoke name="forum_search">'
            )
        )

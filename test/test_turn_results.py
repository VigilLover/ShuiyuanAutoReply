import asyncio
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from shuiyuan_auto_reply.application.tool_results import (
    TurnResults,
    current_turn,
    turn_scope,
)
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel


class TurnResultsTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_flight_and_failure_retry(self):
        turn = TurnResults()

        async def fetch():
            await asyncio.sleep(0)
            return {"status": "ok", "answer": 1}

        mock = AsyncMock(side_effect=fetch)
        result = await asyncio.gather(*(turn.query("same", mock) for _ in range(8)))
        self.assertEqual(len(result), 8)
        mock.assert_awaited_once()
        await turn.query("same", mock)
        mock.assert_awaited_once()
        await turn.query("same", mock, refresh=True)
        self.assertEqual(mock.await_count, 2)
        bad = AsyncMock(return_value={"status": "error"})
        await turn.query("bad", bad)
        await turn.query("bad", bad)
        self.assertEqual(bad.await_count, 2)

    async def test_parallel_turns_and_cleanup(self):
        @turn_scope
        async def run(value):
            turn = current_turn.get()
            turn.cache["user"] = value
            await asyncio.sleep(0)
            return turn.cache["user"]

        self.assertEqual(await asyncio.gather(run("a"), run("b")), ["a", "b"])
        self.assertIsNone(current_turn.get())

    async def test_mixed_valid_invalid_batch_has_one_response_per_call(self):
        async def add(value: int):
            return value + 1

        tool = StructuredTool.from_function(
            coroutine=add, name="add", description="Add one"
        )
        owner = SimpleNamespace(
            tools=[tool],
            _extract_tool_call_name_args=MentionChatModel._extract_tool_call_name_args,
        )
        calls = [
            {"id": "a", "name": "add", "args": {"value": 2}},
            {"id": "b", "name": "add", "args": {}},
            {"id": "c", "name": "missing", "args": {}},
        ]
        state = {"messages": [AIMessage(content="", tool_calls=calls)]}
        state.update(await MentionChatModel._validate_tool_calls(owner, state))
        result = await MentionChatModel._execute_tools(owner, state)
        self.assertEqual([m.tool_call_id for m in result["messages"]], ["a", "b", "c"])
        self.assertEqual(
            [m.status for m in result["messages"]], ["success", "error", "error"]
        )

    async def test_image_tool_batch_uses_image_timeout(self):
        async def generate_image(prompt: str):
            await asyncio.sleep(0.2)
            return {"status": "ok", "prompt": prompt}

        tool = StructuredTool.from_function(
            coroutine=generate_image,
            name="generate_image",
            description="Generate an image",
        )
        owner = SimpleNamespace(tools=[tool])
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "image",
                            "name": "generate_image",
                            "args": {"prompt": "draw a white wolf"},
                        }
                    ],
                )
            ]
        }
        turn = TurnResults()
        turn.control.model_call_timeout = 0.1
        token = current_turn.set(turn)
        try:
            with patch.dict(os.environ, {"IMAGE_GEN_TIMEOUT_SECONDS": "0.5"}):
                result = await MentionChatModel._execute_tools(owner, state)
        finally:
            current_turn.reset(token)

        self.assertEqual(result["messages"][0].status, "success")
        self.assertEqual(turn.control.stop_reason, "")

    async def test_non_image_tool_batch_keeps_model_timeout(self):
        async def slow_tool(value: int):
            await asyncio.sleep(0.2)
            return value

        tool = StructuredTool.from_function(
            coroutine=slow_tool,
            name="slow_tool",
            description="Wait before returning",
        )
        owner = SimpleNamespace(tools=[tool])
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "slow", "name": "slow_tool", "args": {"value": 1}}
                    ],
                )
            ]
        }
        turn = TurnResults()
        turn.control.model_call_timeout = 0.1
        token = current_turn.set(turn)
        try:
            result = await MentionChatModel._execute_tools(owner, state)
        finally:
            current_turn.reset(token)

        self.assertEqual(result["messages"][0].status, "error")
        self.assertEqual(turn.control.stop_reason, "tool_time_budget")

    async def test_read_timeout_keeps_one_generation_only_round(self):
        async def slow_search(query: str):
            await asyncio.sleep(0.2)
            return {"status": "ok", "query": query}

        async def generate_image(prompt: str):
            return {"status": "ok", "image": "[图1]", "prompt": prompt}

        search = StructuredTool.from_function(
            coroutine=slow_search, name="forum_search", description="Search"
        )
        generate = StructuredTool.from_function(
            coroutine=generate_image, name="generate_image", description="Draw"
        )
        owner = SimpleNamespace(tools=[search, generate])
        turn = TurnResults()
        turn.control.image_requested = True
        turn.control.model_call_timeout = 0.05
        turn.control.before_model(turn.progress, turn.deadline)
        token = current_turn.set(turn)
        try:
            timed_out = await MentionChatModel._execute_tools(
                owner,
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "slow",
                                    "name": "forum_search",
                                    "args": {"query": "people"},
                                }
                            ],
                        )
                    ]
                },
            )
            self.assertEqual(timed_out["messages"][0].status, "error")
            self.assertEqual(turn.progress.phase, "investigate")
            self.assertTrue(turn.control.image_nudged)
            self.assertEqual(turn.control.image_nudge_reason, "tool_time_budget")

            turn.control.before_model(turn.progress, turn.deadline)
            result = await MentionChatModel._execute_tools(
                owner,
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "read-again",
                                    "name": "forum_search",
                                    "args": {"query": "more people"},
                                },
                                {
                                    "id": "draw",
                                    "name": "generate_image",
                                    "args": {"prompt": "draw the known people"},
                                },
                            ],
                        )
                    ]
                },
            )
        finally:
            current_turn.reset(token)

        self.assertEqual([m.status for m in result["messages"]], ["error", "success"])
        self.assertEqual(turn.control.queries, 1)
        self.assertEqual(turn.progress.phase, "final")
        self.assertEqual(turn.control.stop_reason, "tool_time_budget")

    async def test_query_limit_rejects_reads_but_allows_requested_image(self):
        async def search(query: str):
            return {"status": "ok", "query": query}

        async def generate_image(prompt: str):
            return {"status": "ok", "image": "[图1]", "prompt": prompt}

        owner = SimpleNamespace(
            tools=[
                StructuredTool.from_function(
                    coroutine=search, name="forum_search", description="Search"
                ),
                StructuredTool.from_function(
                    coroutine=generate_image, name="generate_image", description="Draw"
                ),
            ]
        )
        turn = TurnResults()
        turn.control.image_requested = True
        turn.control.query_limit = 1
        turn.control.queries = 1
        turn.control.before_model(turn.progress, time.monotonic() + 900)
        self.assertTrue(turn.control.image_nudged)
        self.assertEqual(turn.control.image_nudge_reason, "query_budget")
        token = current_turn.set(turn)
        try:
            result = await MentionChatModel._execute_tools(
                owner,
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "read",
                                    "name": "forum_search",
                                    "args": {"query": "more"},
                                },
                                {
                                    "id": "draw",
                                    "name": "generate_image",
                                    "args": {"prompt": "draw the known people"},
                                },
                            ],
                        )
                    ]
                },
            )
        finally:
            current_turn.reset(token)

        self.assertEqual([m.status for m in result["messages"]], ["error", "success"])
        self.assertEqual(turn.control.queries, 1)
        self.assertEqual(turn.progress.phase, "final")
        self.assertEqual(turn.control.stop_reason, "query_budget")

    async def test_image_tool_batch_still_honors_image_timeout(self):
        async def generate_image(prompt: str):
            await asyncio.sleep(0.2)
            return {"status": "ok", "prompt": prompt}

        tool = StructuredTool.from_function(
            coroutine=generate_image,
            name="generate_image",
            description="Generate an image",
        )
        owner = SimpleNamespace(tools=[tool])
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "image",
                            "name": "generate_image",
                            "args": {"prompt": "draw a white wolf"},
                        }
                    ],
                )
            ]
        }
        turn = TurnResults()
        turn.control.model_call_timeout = 0.5
        token = current_turn.set(turn)
        try:
            with patch.dict(os.environ, {"IMAGE_GEN_TIMEOUT_SECONDS": "0.1"}):
                result = await MentionChatModel._execute_tools(owner, state)
        finally:
            current_turn.reset(token)

        self.assertEqual(result["messages"][0].status, "error")
        self.assertEqual(turn.control.stop_reason, "tool_time_budget")

    def test_result_snapshots_are_immutable_and_deduplicated(self):
        turn = TurnResults()
        source = {"users": ["Alice"]}
        first = turn.save(source)
        self.assertEqual(first, turn.save(source))
        source["users"].append("Bob")
        self.assertNotIn("Bob", turn.read(first)["content"])
        self.assertNotEqual(first, turn.save(source))

    async def test_nonretryable_read_failure_stops_at_first_attempt(self):
        from shuiyuan_auto_reply.application.tool_results import tool_error
        from shuiyuan_auto_reply.domain.tool_error import ReadFailure
        from shuiyuan_auto_reply.retry import async_retry

        call = AsyncMock(side_effect=ReadFailure(403))
        with self.assertRaises(ReadFailure):
            await async_retry()(call)()
        call.assert_awaited_once()
        self.assertEqual(tool_error(ReadFailure(429))["error"], "rate_limited")
        self.assertTrue(tool_error(ReadFailure(429))["retryable"])

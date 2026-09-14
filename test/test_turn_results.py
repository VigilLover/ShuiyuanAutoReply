import asyncio
import unittest
from unittest.mock import AsyncMock

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
        from types import SimpleNamespace

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

"""Regression: a tool call must keep its own result across message-list merges.

A repeated read is answered from this turn's cache. The replay used to reuse the
stored message id, so the graph's message merge replaced the earlier message in
place and the next provider request carried a call without a result — which
OpenAI-compatible Responses endpoints reject with HTTP 400, failing the turn.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from test_forum_agent_flow import OfflineChat

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.context_budget import repair_tool_pairing


def pairing(messages) -> dict[str, bool]:
    """Map each tool call id to whether a result for it is present."""
    call_ids = []
    answered = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            call_ids.append(call["id"])
        if isinstance(message, ToolMessage) and message.tool_call_id:
            answered.add(message.tool_call_id)
    return {call_id: call_id in answered for call_id in call_ids}


class RepairToolPairingTests(unittest.TestCase):
    def test_healthy_conversation_is_untouched(self):
        messages = [
            AIMessage(
                content="", tool_calls=[{"id": "c1", "name": "get_post", "args": {}}]
            ),
            ToolMessage(content="ok", tool_call_id="c1", name="get_post"),
        ]
        repaired, changed = repair_tool_pairing(messages)
        self.assertEqual(changed, [])
        self.assertEqual(repaired, messages)

    def test_missing_result_is_filled_right_after_its_call(self):
        messages = [
            AIMessage(
                content="", tool_calls=[{"id": "c1", "name": "get_post", "args": {}}]
            ),
            HumanMessage(content="继续"),
        ]
        repaired, changed = repair_tool_pairing(messages)
        self.assertEqual(changed, ["c1"])
        self.assertIsInstance(repaired[1], ToolMessage)
        self.assertEqual(repaired[1].tool_call_id, "c1")
        self.assertEqual(repaired[1].status, "error")

    def test_result_without_its_call_is_dropped(self):
        orphan = ToolMessage(content="ok", tool_call_id="gone", name="get_post")
        messages = [
            AIMessage(
                content="", tool_calls=[{"id": "c1", "name": "get_post", "args": {}}]
            ),
            ToolMessage(content="ok", tool_call_id="c1", name="get_post"),
            orphan,
        ]
        repaired, changed = repair_tool_pairing(messages)
        self.assertEqual(changed, [])
        self.assertNotIn(orphan, repaired)
        self.assertEqual(pairing(repaired), {"c1": True})

    def test_batched_calls_are_matched_by_id_not_by_position(self):
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "get_post", "args": {}},
                    {"id": "c2", "name": "search_user", "args": {}},
                ],
            ),
            ToolMessage(content="ok", tool_call_id="c2", name="search_user"),
        ]
        repaired, changed = repair_tool_pairing(messages)
        self.assertEqual(changed, ["c1"])
        self.assertEqual(pairing(repaired), {"c1": True, "c2": True})


class RepeatedReadTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_read_keeps_every_call_answered(self):
        model = SimpleNamespace(
            get_user_by_username=AsyncMock(
                return_value=SimpleNamespace(
                    id=9, username="Alice", name=None, avatar_template=None
                )
            )
        )
        runtime = OfflineChat(model)
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            messages = [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "get_user",
                            "args": {"username": "Alice"},
                        }
                    ],
                )
            ]
            first = await runtime._execute_tools({"messages": messages})
            messages = add_messages(messages, first["messages"])
            messages = add_messages(
                messages,
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-2",
                                "name": "get_user",
                                "args": {"username": "Alice"},
                            }
                        ],
                    )
                ],
            )
            second = await runtime._execute_tools({"messages": messages})
            messages = add_messages(messages, second["messages"])
            # The replay is a new message, not a replacement of the earlier one.
            self.assertNotEqual(first["messages"][0].id, second["messages"][0].id)
            self.assertEqual(pairing(messages), {"call-1": True, "call-2": True})
            model.get_user_by_username.assert_awaited_once_with("Alice")
        finally:
            current_turn.reset(token)

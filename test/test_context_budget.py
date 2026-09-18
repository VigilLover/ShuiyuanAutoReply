"""Stable-prefix projection: old rounds are dropped whole, kept rounds untouched."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.context_budget import project_messages


def _round(index: int, size: int = 400) -> list:
    return [
        AIMessage(
            content="",
            tool_calls=[
                {"id": f"call-{index}", "name": "forum_read", "args": {"n": index}}
            ],
        ),
        ToolMessage(
            content=f"round {index} " + ("evidence " * size),
            tool_call_id=f"call-{index}",
            name="forum_read",
        ),
    ]


class StablePrefixProjectionTests(unittest.TestCase):
    def setUp(self):
        self.turn = TurnResults()
        self.token = current_turn.set(self.turn)

    def tearDown(self):
        current_turn.reset(self.token)

    def test_over_budget_drops_oldest_rounds_and_keeps_the_rest_verbatim(self):
        messages = [HumanMessage(content="task")]
        for index in range(12):
            messages += _round(index)
        projected = project_messages(messages, 3000)

        kept_ids = [m.tool_call_id for m in projected if isinstance(m, ToolMessage)]
        # Newest three rounds always survive; everything kept is the original object.
        self.assertTrue({"call-9", "call-10", "call-11"} <= set(kept_ids))
        self.assertLess(len(kept_ids), 12)
        originals = {id(m) for m in messages}
        self.assertTrue(all(id(m) in originals for m in projected))
        # Dropped rounds were archived for the finalizer.
        self.assertGreaterEqual(len(self.turn.results), 12 - len(kept_ids))

    def test_projection_is_a_prefix_of_the_previous_projection_plus_new_round(self):
        messages = [HumanMessage(content="task")]
        for index in range(10):
            messages += _round(index)
        first = project_messages(messages, 3000)
        messages += _round(10)
        second = project_messages(messages, 3000)

        def keys(projected):
            return [
                (
                    m.tool_calls[0]["id"]
                    if isinstance(m, AIMessage) and m.tool_calls
                    else getattr(m, "tool_call_id", None) or m.content
                )
                for m in projected
            ]

        first_ids, second_ids = keys(first), keys(second)
        # The next request reuses the previous projection as its prefix (modulo
        # rounds dropped from the front), so the provider cache keeps hitting.
        overlap = [item for item in first_ids if item in second_ids]
        self.assertEqual(overlap, second_ids[: len(overlap)])
        self.assertIn("call-10", second_ids)

    def test_within_budget_returns_input_unchanged(self):
        messages = [HumanMessage(content="task")] + _round(0, size=5)
        self.assertEqual(project_messages(messages, 3000), messages)

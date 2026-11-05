"""Stable-prefix projection: old rounds are dropped whole, kept rounds untouched."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.context_budget import project_messages
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.shuiyuan.objects import PostDetails


def _round(index: int, size: int = 400, width: int = 1) -> list:
    calls = [
        {
            "id": f"call-{index}" if width == 1 else f"call-{index}-{item}",
            "name": "forum_read",
            "args": {"n": index},
        }
        for item in range(width)
    ]
    return [
        AIMessage(content="", tool_calls=calls),
        *[
            ToolMessage(
                content=f"round {index} " + ("evidence " * size),
                tool_call_id=call["id"],
                name="forum_read",
            )
            for call in calls
        ],
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
        for rounds, width in ((1, 1), (10, 1), (4, 4)):
            with self.subTest(rounds=rounds, width=width):
                messages = [HumanMessage(content="task")]
                for index in range(rounds):
                    messages += _round(index, size=5, width=width)
                self.assertEqual(project_messages(messages, 3000), messages)
                self.assertEqual(
                    MentionChatModel._trim_tool_loop_messages(messages), messages
                )


def _post_details(raw: str = "hello") -> PostDetails:
    return PostDetails(
        id=1,
        name="奇诺",
        user_id=113224,
        username="Kino",
        user_cakedate=None,
        created_at="2026-01-01T00:00:00Z",
        cooked=f"<p>{raw}</p>",
        raw=raw,
        post_number=7,
        post_type=1,
        updated_at="2026-01-01T00:00:00Z",
        reply_count=0,
        reply_to_post_number=None,
        reply_to_user=None,
        polls=None,
        yours=False,
        topic_id=99,
        can_edit=False,
        can_delete=False,
        can_recover=False,
        can_wiki=False,
        can_retort=False,
        can_remove_retort=False,
        can_accept_answer=False,
        can_unaccept_answer=False,
        can_see_hidden_post=False,
        can_view_edit_history=False,
    )


def _assert_valid_tool_sequence(testcase, messages):
    """Assert assistant tool_calls are always followed by their complete ToolMessages."""
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            expected_ids = {
                call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                for call in message.tool_calls
            }
            index += 1
            actual_ids = set()
            while index < len(messages) and isinstance(messages[index], ToolMessage):
                actual_ids.add(messages[index].tool_call_id)
                index += 1
            testcase.assertEqual(actual_ids, expected_ids)
        else:
            index += 1


class EvidenceProjectionTests(unittest.TestCase):
    def test_large_parallel_batch_keeps_all_responses_without_public_result_ids(self):
        from shuiyuan_auto_reply.application.tool_results import (
            TurnResults,
            current_turn,
        )
        from shuiyuan_auto_reply.features.mention.context_budget import project_messages

        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            turn.cache["user:alice"] = {
                "username": "alice",
                "avatar": "https://example.org/a.png",
            }
            calls = [
                {"name": "get_user", "args": {"username": str(i)}, "id": str(i)}
                for i in range(30)
            ]
            messages = [
                HumanMessage(content="Use the supplied users"),
                AIMessage(content="", tool_calls=calls),
            ]
            messages += [
                ToolMessage(content=(f"evidence-{i} " * 3000), tool_call_id=str(i))
                for i in range(30)
            ]
            projected = project_messages(messages, 4000)
            _assert_valid_tool_sequence(self, projected)
            self.assertEqual(sum(isinstance(m, ToolMessage) for m in projected), 30)
            self.assertFalse(
                any("result_id" in str(message.content) for message in projected)
            )
            self.assertEqual(
                turn.cache["user:alice"]["avatar"], "https://example.org/a.png"
            )
        finally:
            current_turn.reset(token)

    def test_target_and_latest_two_tool_batches_survive_long_history(self):
        from shuiyuan_auto_reply.application.tool_results import (
            TurnResults,
            current_turn,
        )
        from shuiyuan_auto_reply.features.mention.context_budget import project_messages

        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            messages = [
                HumanMessage(content="current task"),
                HumanMessage(content="target raw body", name="target_post"),
            ]
            for i in range(107):
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": str(i),
                                "name": "get_post",
                                "args": {"topic_id": 42, "post_number": i + 1},
                            }
                        ],
                    )
                )
                messages.append(
                    ToolMessage(
                        content=f"post {i} " + "long content " * 1000,
                        tool_call_id=str(i),
                    )
                )
            projected = project_messages(messages, 4000)
            _assert_valid_tool_sequence(self, projected)
            self.assertTrue(any(m.name == "target_post" for m in projected))
            ids = {m.tool_call_id for m in projected if isinstance(m, ToolMessage)}
            self.assertTrue({"105", "106"} <= ids)
            self.assertFalse(any("results_index" in str(m.content) for m in projected))
        finally:
            current_turn.reset(token)

    def test_projection_does_not_inject_cached_objects_into_model_context(self):
        from shuiyuan_auto_reply.application.tool_results import (
            TurnResults,
            current_turn,
        )
        from shuiyuan_auto_reply.features.mention.context_budget import project_messages
        from shuiyuan_auto_reply.features.mention.shuiyuan_tools_objects import (
            PostShort,
        )

        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            # get_post/get_post_by_id cache PostShort objects under post: keys.
            turn.cache["post:1"] = PostShort(
                _post_details(raw="我也很喜欢这首歌"), "随性更日记", full=True
            )
            messages = [HumanMessage(content="判断楼主喜欢什么歌")]
            for i in range(60):
                messages.append(
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": str(i),
                                "name": "get_post",
                                "args": {"topic_id": 42, "post_number": i + 1},
                            }
                        ],
                    )
                )
                messages.append(
                    ToolMessage(
                        content="post body " * 400, tool_call_id=str(i), name="get_post"
                    )
                )
            projected = project_messages(messages, 4000)
            _assert_valid_tool_sequence(self, projected)
            self.assertFalse(
                any("known_entities" in str(message.content) for message in projected)
            )
        finally:
            current_turn.reset(token)

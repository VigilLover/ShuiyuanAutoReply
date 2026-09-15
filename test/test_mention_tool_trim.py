import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.shuiyuan.objects import PostDetails


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


class MentionToolLoopTrimTests(unittest.TestCase):
    def test_trim_does_not_split_parallel_tool_call_batch(self):
        messages = [HumanMessage(content="hi")]
        for i in range(4):
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": f"f{i}",
                            "args": {},
                            "id": f"a{i}_{j}",
                            "type": "tool_call",
                        }
                        for j in range(4)
                    ],
                )
            )
            for j in range(4):
                messages.append(
                    ToolMessage(
                        content=f"res {i}_{j}",
                        tool_call_id=f"a{i}_{j}",
                        name=f"f{i}",
                    )
                )

        self.assertEqual(len(messages), 21)
        trimmed = MentionChatModel._trim_tool_loop_messages(messages)

        self.assertEqual(
            trimmed, messages
        )  # Small results no longer hit a message-count cap.
        _assert_valid_tool_sequence(self, trimmed)

    def test_trim_single_tool_calls_still_valid(self):
        messages = [HumanMessage(content="hi")]
        for i in range(10):
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": f"f{i}",
                            "args": {},
                            "id": f"a{i}",
                            "type": "tool_call",
                        }
                    ],
                )
            )
            messages.append(
                ToolMessage(content=f"res {i}", tool_call_id=f"a{i}", name=f"f{i}")
            )

        self.assertEqual(len(messages), 21)
        trimmed = MentionChatModel._trim_tool_loop_messages(messages)

        self.assertEqual(
            trimmed, messages
        )  # Small results no longer hit a message-count cap.
        _assert_valid_tool_sequence(self, trimmed)

    def test_trim_returns_original_when_within_limit(self):
        messages = [HumanMessage(content="hi")]
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "f", "args": {}, "id": "a0", "type": "tool_call"}],
            )
        )
        messages.append(ToolMessage(content="res", tool_call_id="a0", name="f"))

        trimmed = MentionChatModel._trim_tool_loop_messages(messages)
        self.assertEqual(trimmed, messages)


if __name__ == "__main__":
    unittest.main()


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

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel


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

        self.assertLessEqual(len(trimmed), MentionChatModel._MAX_TOOL_LOOP_MESSAGES)
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

        self.assertLessEqual(len(trimmed), MentionChatModel._MAX_TOOL_LOOP_MESSAGES)
        _assert_valid_tool_sequence(self, trimmed)

    def test_trim_returns_original_when_within_limit(self):
        messages = [HumanMessage(content="hi")]
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "f", "args": {}, "id": "a0", "type": "tool_call"}
                ],
            )
        )
        messages.append(
            ToolMessage(content="res", tool_call_id="a0", name="f")
        )

        trimmed = MentionChatModel._trim_tool_loop_messages(messages)
        self.assertEqual(trimmed, messages)


if __name__ == "__main__":
    unittest.main()

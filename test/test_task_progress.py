from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from shuiyuan_auto_reply.application.tool_results import TurnResults, current_turn
from shuiyuan_auto_reply.features.mention.context_budget import project_messages


def test_sources_deduplicate_without_model_written_scope_ledger():
    turn = TurnResults()
    post = {
        "post_id": 42,
        "topic_id": 3,
        "author": {"username": "Alice"},
        "content": "likes music",
    }
    first = turn.observe({"items": [post]}, tool="forum_search")
    assert len(first) == 1
    assert not turn.observe({"items": [post]}, tool="forum_search")
    key = next(iter(first))
    assert turn.read(key, field="content")["content"] == "likes music"
    assert turn.read(key, field="missing")["status"] == "error"


def test_readback_survives_projection_and_index_is_not_recursive():
    turn = TurnResults()
    token = current_turn.set(turn)
    try:
        turn.observe({"post_id": 42, "content": "evidence"})
        messages = [
            HumanMessage(content="goal"),
            AIMessage(
                content="",
                tool_calls=[{"id": "a", "name": "forum_read", "args": {}}],
            ),
            ToolMessage(content="x" * 1000, tool_call_id="a", name="forum_read"),
        ]
        messages.insert(1, HumanMessage(content="long " * 5000))
        first = project_messages(messages, 2000)
        second = project_messages(messages, 2000)
        assert (
            next(m for m in second if isinstance(m, ToolMessage)).content == "x" * 1000
        )
        assert len(turn.evidence) == 1
        assert all(
            "results_index" not in item["preview"] for item in turn.evidence.values()
        )
    finally:
        current_turn.reset(token)

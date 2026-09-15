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


def test_full_post_upgrades_summary_once_by_canonical_reference():
    turn = TurnResults()
    summary = {
        "ref": "forum:42/7",
        "author": "Alice",
        "text": "命中摘要",
    }
    full = {
        "ref": "forum:42/7",
        "post_id": 70,
        "author": "Alice",
        "text": "完整正文",
    }
    assert turn.observe({"items": [summary]}, tool="forum_search") == {"forum:42/7"}
    assert turn.observe({"items": [full]}, tool="forum_read") == {"forum:42/7"}
    assert not turn.observe(
        {"items": [{**full, "text": "完整正文（格式变化）"}]}, tool="forum_read"
    )
    assert len(turn.evidence) == 1
    assert turn.evidence["forum:42/7"]["kind"] == "full"
    assert turn.read("forum:42/7", field="text")["content"] == "完整正文"


def test_completed_topic_detection_keeps_filtered_and_exact_reads_available():
    turn = TurnResults()
    turn.completed_topics.add(42)
    assert turn.is_redundant_completed_call("forum_read", {"topic_id": 42})
    assert turn.is_redundant_completed_call(
        "forum_search", {"query": "topic:42 order:latest"}
    )
    assert not turn.is_redundant_completed_call(
        "forum_search", {"query": "关键词", "topic_id": 42}
    )
    assert not turn.is_redundant_completed_call(
        "forum_read", {"topic_id": 42, "post_number": 7}
    )


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

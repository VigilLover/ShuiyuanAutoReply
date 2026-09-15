import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from test_forum_agent_flow import OfflineChat

from shuiyuan_auto_reply.application.retrieval_control import RetrievalControl
from shuiyuan_auto_reply.application.task_progress import TaskProgress
from shuiyuan_auto_reply.shuiyuan.objects import User


def test_three_empty_batches_stop_without_model_written_review():
    p = TaskProgress()
    c = RetrievalControl()
    for _ in range(3):
        c.after_batch(p, new_evidence=0, reads=1)
    assert p.phase == "final"
    assert c.stop_reason == "no_new_evidence"


def test_budget_reserves_last_model_request():
    c = RetrievalControl(model_limit=3)
    p = TaskProgress()
    for _ in range(3):
        c.before_model(p, time.monotonic() + 900)
    assert p.phase == "final"
    assert c.model_rounds == 3


def test_real_graph_stops_model_ignoring_review_and_reuses_read():
    asyncio.run(_real_graph_stops_model_ignoring_review_and_reuses_read())


async def _real_graph_stops_model_ignoring_review_and_reuses_read():
    post = SimpleNamespace(
        id=10,
        topic_id=42,
        post_number=1,
        reply_to_post_number=None,
        user_id=1,
        username="Alice",
        name="Alice",
        raw="favourite song",
        cooked="<p>song</p>",
    )
    model = SimpleNamespace(
        get_post_details_by_post_number=AsyncMock(return_value=post),
    )
    runtime = OfflineChat(model)
    runtime.llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="资料有限，已停止查询。"))
    )
    calls = []

    async def repeat(prompt):
        calls.append(prompt)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": str(len(calls)),
                    "name": "forum_read",
                    "args": {"topic_id": 42, "post_number": 1},
                }
            ],
        )

    runtime.llm_with_tools = SimpleNamespace(ainvoke=repeat)
    result = await runtime.get_pumpkin_response(
        topic_id=42,
        reply_to_post_number=None,
        conversation="read main post",
        user=User(id=2, username="requester", name=""),
    )
    assert "已停止" in result
    assert len(calls) <= 5
    assert model.get_post_details_by_post_number.await_count == 1
    runtime.llm.ainvoke.assert_awaited_once()

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


def test_custom_call_timeout_cap_still_reserves_final_response(monkeypatch):
    monkeypatch.setattr(time, "monotonic", lambda: 1000)
    control = RetrievalControl(
        model_call_timeout=180,
        final_reserve_seconds=150,
    )

    assert control.call_timeout(1900, final=False) == 180
    assert control.call_timeout(1900, final=False, max_seconds=600) == 600
    assert control.call_timeout(1600, final=False, max_seconds=600) == 450


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


def test_requested_image_gets_one_generation_round_before_evidence_stop():
    p = TaskProgress()
    c = RetrievalControl(image_requested=True)
    c.stop_investigation(p, "source_complete")
    assert p.phase == "investigate"
    assert c.image_nudged
    c.before_model(p, time.monotonic() + 900)
    c.stop_investigation(p, "source_complete")
    assert p.phase == "final"
    assert c.stop_reason == "source_complete"


def test_both_evidence_checks_in_one_batch_keep_the_generation_round():
    p = TaskProgress()
    c = RetrievalControl(image_requested=True, no_progress=1)
    c.before_model(p, time.monotonic() + 900)
    c.after_batch(p, new_evidence=0, reads=1)
    c.stop_investigation(p, "source_complete")
    assert p.phase == "investigate"
    c.before_model(p, time.monotonic() + 900)
    c.stop_investigation(p, "source_complete")
    assert p.phase == "final"


def test_evidence_stop_is_immediate_once_image_was_attempted():
    p = TaskProgress()
    c = RetrievalControl(image_requested=True, image_attempted=True)
    for _ in range(2):
        c.after_batch(p, new_evidence=0, reads=1)
    assert p.phase == "final"
    assert not c.image_nudged


def test_budget_stops_ignore_image_nudge():
    c = RetrievalControl(model_limit=2, image_requested=True)
    p = TaskProgress()
    c.before_model(p, time.monotonic() + 900)
    c.before_model(p, time.monotonic() + 900)
    assert p.phase == "final"
    assert c.stop_reason == "model_budget"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("你能按照主楼的提示词加工一下我的头像生成图片吗", True),
        ("给我的头像也加一个背景吧", True),
        ("能试试看在这个的基础上加上原头像里的蓝天背景吗", True),
        ("帮我画一只猫", True),
        ("帮我P图", True),
        ("PDF 的背景怎么设置", False),
        ("你的图呢给我吐出来。", False),
        ("今天食堂吃什么", False),
    ],
)
def test_image_request_detection(text, expected):
    from shuiyuan_auto_reply.features.mention.image_intent import wants_image

    assert wants_image(text) is expected


def test_final_image_notice_reflects_turn_images():
    from shuiyuan_auto_reply.application.tool_results import TurnResults
    from shuiyuan_auto_reply.features.mention.finalize import FinalizeMixin

    turn = TurnResults()
    assert "本轮没有生成或选取任何图片" in FinalizeMixin._final_image_notice(turn)
    assert "本轮没有生成" in FinalizeMixin._final_image_notice(None)
    turn.images.register(SimpleNamespace(artifact_id="a"), "本轮生成图")
    notice = FinalizeMixin._final_image_notice(turn)
    assert "[图1]（本轮生成图）" in notice
    assert "://" not in notice

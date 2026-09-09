"""Offline checks for filtered, concurrent, durable forum monitoring."""

import asyncio
import json
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from shuiyuan_auto_reply.application import BotService, HandlerRegistry
from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.application.handlers import CallbackChatHandler
from shuiyuan_auto_reply.domain import ReplyResult
from shuiyuan_auto_reply.features.mention.mention_model import MentionModel
from shuiyuan_auto_reply.infrastructure.forum import ForumOutputFormatter
from shuiyuan_auto_reply.infrastructure.persistence import (
    SQLiteSessionRepository,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.infrastructure.persistence.work_queue import (
    ForumQueue,
    _current_job,
    publication_status,
)
from shuiyuan_auto_reply.shuiyuan.objects import UserActionDetails
from shuiyuan_auto_reply.shuiyuan.user_action_model import BaseUserActionModel


def action(post_id=1, topic=10):
    values = {f.name: 0 for f in fields(UserActionDetails)}
    for name in (
        "excerpt",
        "created_at",
        "avatar_template",
        "acting_avatar_template",
        "slug",
        "target_name",
        "target_username",
        "username",
        "name",
        "acting_username",
        "acting_name",
        "title",
    ):
        values[name] = "test"
    for name in ("deleted", "hidden", "closed", "archived"):
        values[name] = False
    return UserActionDetails(
        **{
            **values,
            "post_id": post_id,
            "post_number": post_id,
            "topic_id": topic,
            "action_type": 7,
        }
    )


async def setup(tmp_path, callback=None):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    await store.initialize()
    model = MentionModel.__new__(MentionModel)

    async def post(post_id):
        return SimpleNamespace(
            id=post_id,
            topic_id=10,
            post_number=post_id,
            reply_to_post_number=None,
            user_id=20,
            username="alice",
            name="Alice",
            raw="@bot 【小狼】hello",
        )

    async def reply(*args):
        await publication_status("sent", 100)

    forum = SimpleNamespace(
        get_post_details=AsyncMock(side_effect=post),
        get_topic_details=AsyncMock(return_value=SimpleNamespace(title="Test topic")),
        reply_to_post=AsyncMock(side_effect=reply),
        get_actions=AsyncMock(return_value=SimpleNamespace(user_actions=[])),
    )
    BaseUserActionModel.__init__(model, forum, "bot", [7])
    model.persona = "wolf_lumine"
    model.state_store = store
    model.runtime_refresher = None
    model.output_formatter = ForumOutputFormatter()

    async def default(context):
        await emit_event(
            "tool.started", {"name": "offline", "arguments": context.request.content}
        )
        return ReplyResult("reply:" + context.request.content)

    model.bot_service = BotService(
        SQLiteSessionRepository(store),
        HandlerRegistry(
            [CallbackChatHandler(lambda text: "【小狼】" in text, callback or default)]
        ),
    )
    return model, store


def test_unmatched_messages_leave_no_monitor_records(tmp_path):
    async def run():
        model, store = await setup(tmp_path)
        from shuiyuan_auto_reply.constants import settings

        for text, user in [
            ("ordinary", "alice"),
            ("@bot 【小狼】ignored" + settings.auto_reply_tag, "bot"),
            ("@bot hello", "alice"),
            ("【小狼】hello", "alice"),
        ]:
            model.model.get_post_details.return_value = None

            async def post(_):
                return SimpleNamespace(
                    raw=text,
                    username=user,
                    id=1,
                    user_id=20,
                    name="A",
                    topic_id=10,
                    post_number=1,
                    reply_to_post_number=None,
                )

            model.model.get_post_details.side_effect = post
            await model._new_action_routine(action())
        assert await store.list_conversations() == []
        assert (await store.forum_monitor())["cursor"] == 0
        model.model.reply_to_post.assert_not_awaited()

    asyncio.run(run())


def test_acceptance_steps_and_publication_are_visible_before_completion(tmp_path):
    async def run():
        entered, release, publishing, publish_release = [
            asyncio.Event() for _ in range(4)
        ]

        async def callback(context):
            await emit_event("tool.started", {"name": "blocked"})
            entered.set()
            await release.wait()
            return ReplyResult("answer")

        model, store = await setup(tmp_path, callback)

        async def publish(*args):
            publishing.set()
            await publish_release.wait()

        model.model.reply_to_post.side_effect = publish
        prepared = await model._accept_action(await model._prepare_action(action()))
        snapshot = await store.forum_monitor()
        assert snapshot["runs"][0]["status"] == "queued"
        assert snapshot["runs"][0]["request"]["content"].startswith("@bot")
        assert await store.list_messages(snapshot["runs"][0]["conversation_id"]) == []
        task = asyncio.create_task(model._execute_action(prepared))
        await entered.wait()
        events = await store.forum_events_after(snapshot["cursor"])
        assert "tool.started" in [e["type"] for e in events]
        assert len({e["run_id"] for e in events}) == 1
        release.set()
        await publishing.wait()
        cid = snapshot["runs"][0]["conversation_id"]
        assert (await store.list_forum_runs(cid))[0]["status"] == "publishing"
        assert "run.completed" not in [
            e["type"] for e in await store.forum_events_after(0)
        ]
        publish_release.set()
        await task
        assert (await store.list_forum_runs(cid))[0]["status"] == "completed"
        assert [m.role for m in await store.list_messages(cid)] == ["user", "assistant"]
        types = [e["type"] for e in await store.forum_events_after(0)]
        assert types.index("forum.reply_published") < types.index("run.completed")

    asyncio.run(run())


def test_matched_lookup_error_is_a_failed_run_even_when_error_reply_sent(tmp_path):
    async def run():
        async def fail(context):
            raise LookupError("tool lookup failed")

        model, store = await setup(tmp_path, fail)
        await model._new_action_routine(action())
        cid = (await store.list_conversations())[0].id
        runs = await store.list_forum_runs(cid)
        assert len(runs) == 1 and runs[0]["status"] == "failed"
        assert "tool lookup failed" in runs[0]["error"]
        model.model.reply_to_post.assert_awaited_once()

    asyncio.run(run())


def test_uncertain_publication_and_recovery(tmp_path):
    async def run():
        model, store = await setup(tmp_path)
        queue = ForumQueue(
            store.path if hasattr(store, "path") else tmp_path / "state.sqlite3", "bot"
        )
        await queue.initialize()
        await queue.enqueue([action()], 1)
        token = _current_job.set((queue, 1))

        async def uncertain(*args):
            await publication_status("sending")
            raise ConnectionError("lost response")

        model.model.reply_to_post.side_effect = uncertain
        try:
            with pytest.raises(ConnectionError):
                await model._new_action_routine(action())
        finally:
            _current_job.reset(token)
        cid = (await store.list_conversations())[0].id
        assert (await store.list_forum_runs(cid))[0]["status"] == "needs_review"
        await queue.initialize()
        await store.recover_forum_runs("bot")
        assert await queue.state(1) == "needs_review"
        assert await queue.pending() == []
        # An interrupted pre-publication attempt is linked to the next attempt.
        prepared = await model._accept_action(await model._prepare_action(action(2)))
        await queue.enqueue([action(2)], 2)
        await store.recover_forum_runs("bot")
        again = await model._accept_action(await model._prepare_action(action(2)))
        runs = await store.list_forum_runs(cid)
        old = next(r for r in runs if r["id"] == prepared["observer"].run_id)
        new = next(r for r in runs if r["id"] == again["observer"].run_id)
        assert old["status"] == "interrupted"
        assert new["request"]["previous_run_id"] == old["id"]

    asyncio.run(run())


def test_worker_prepares_queued_requests_and_preserves_topic_order(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SHUIYUAN_STATE_DIR", str(tmp_path))

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        order = []

        async def callback(context):
            order.append(context.request.forum_context.post_id)
            await emit_event("tool.started", {"name": context.request.request_id})
            if len(order) == 1:
                entered.set()
                await release.wait()
            return ReplyResult("done")

        model, store = await setup(tmp_path, callback)
        queue = ForumQueue(tmp_path / "state.sqlite3", "bot")
        await queue.initialize()
        await queue.enqueue([action(1), action(2)], 2)
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

        original = get_deployment()

        class Config:
            def section(self, name):
                data = original.section(name)
                return {**data, "poll_interval": 0.01} if name == "runtime" else data

        with patch(
            "shuiyuan_auto_reply.bootstrap.deployment.get_deployment",
            return_value=Config(),
        ):
            worker = asyncio.create_task(model.watch_new_action_routine())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                for _ in range(100):
                    snapshot = await store.forum_monitor()
                    if len(snapshot["runs"]) == 2:
                        break
                    await asyncio.sleep(0.01)
                assert sorted(r["status"] for r in snapshot["runs"]) == [
                    "queued",
                    "running",
                ]
                assert order == [1]
                release.set()
                for _ in range(100):
                    if await queue.state(2) == "sent":
                        break
                    await asyncio.sleep(0.01)
                assert order == [1, 2]
            finally:
                release.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                await BaseUserActionModel.aclose(model)

    asyncio.run(run())


def test_monitor_api_and_sse_cursor(tmp_path):
    async def run():
        from shuiyuan_auto_reply.interfaces.api.app import create_app

        model, store = await setup(tmp_path)
        prepared = await model._accept_action(await model._prepare_action(action()))
        snapshot = await store.forum_monitor()
        await model._execute_action(prepared)
        app = create_app()
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(container=SimpleNamespace(state_store=store))
            ),
            headers={"last-event-id": str(snapshot["cursor"])},
            is_disconnected=AsyncMock(return_value=False),
        )
        endpoint = next(
            r.endpoint
            for r in app.routes
            if getattr(r, "path", "") == "/api/forum/events/stream"
        )
        response = await endpoint(request, 0)
        iterator = response.body_iterator
        first = await anext(iterator)
        payload = json.loads(first.split("data: ")[1])
        assert payload["event_id"] > snapshot["cursor"]
        assert payload["conversation_id"] == prepared["observer"].conversation_id
        await iterator.aclose()
        await store.transition_forum_run(
            prepared["observer"].run_id, "failed", "run.failed"
        )
        assert (await store.list_forum_runs(prepared["observer"].conversation_id))[0][
            "status"
        ] == "completed"

    asyncio.run(run())


def test_three_topics_execute_concurrently_with_isolated_events(tmp_path):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        active = 0

        async def callback(context):
            nonlocal active
            active += 1
            await emit_event("tool.started", {"name": context.request.request_id})
            if active == 3:
                entered.set()
            await release.wait()
            return ReplyResult("done")

        model, store = await setup(tmp_path, callback)
        original = model.model.get_post_details.side_effect

        async def post(post_id):
            result = await original(post_id)
            result.topic_id = post_id
            return result

        model.model.get_post_details.side_effect = post
        tasks = [
            asyncio.create_task(model._new_action_routine(action(i, i)))
            for i in range(1, 5)
        ]
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert active == 3
            snapshot = await store.forum_monitor()
            assert sum(r["status"] == "running" for r in snapshot["runs"]) == 3
            for r in snapshot["runs"]:
                events = await store.list_events_for_conversation(r["conversation_id"])
                tools = [e for e in events if e.event_type == "tool.started"]
                assert all(e.payload["name"] == r["request_id"] for e in tools)
        finally:
            release.set()
            await asyncio.gather(*tasks)

    asyncio.run(run())


def test_cancellation_and_timeout_finish_monitor_runs(tmp_path):
    async def run():
        entered = asyncio.Event()

        async def block(context):
            entered.set()
            await asyncio.Event().wait()

        model, store = await setup(tmp_path, block)
        task = asyncio.create_task(model._new_action_routine(action()))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cid = (await store.list_conversations())[0].id
        assert (await store.list_forum_runs(cid))[0]["status"] == "interrupted"
        from shuiyuan_auto_reply.application.scheduling import ReplyScheduler

        with patch(
            "shuiyuan_auto_reply.application.scheduling.get_scheduler",
            return_value=ReplyScheduler(3, 100, 0.03),
        ):
            with pytest.raises(TimeoutError):
                await model._new_action_routine(action(2))
        statuses = [r["status"] for r in await store.list_forum_runs(cid)]
        assert statuses == ["interrupted", "failed"]
        model.model.reply_to_post.assert_not_awaited()

    asyncio.run(run())


def test_clear_keeps_trace_without_polluting_history(tmp_path):
    async def run():
        from shuiyuan_auto_reply.application.handlers import ClearHandler

        model, store = await setup(tmp_path)
        callback = AsyncMock(return_value=ReplyResult("已清除"))
        model.bot_service = BotService(
            SQLiteSessionRepository(store),
            HandlerRegistry([ClearHandler(lambda text: True, callback)]),
        )
        await model._new_action_routine(action())
        cid = (await store.list_conversations())[0].id
        events = await store.list_events_for_conversation(cid)
        assert [e.event_type for e in events][0] == "run.accepted"
        assert any(
            e.event_type == "run.generated" and e.payload["text"] == "已清除"
            for e in events
        )
        assert (await store.list_forum_runs(cid))[0]["status"] == "completed"
        assert all(m.role == "system" for m in await store.list_messages(cid))

    asyncio.run(run())


def test_restart_reconciles_sent_and_failed_generation(tmp_path):
    async def run():
        model, store = await setup(tmp_path)
        queue = ForumQueue(tmp_path / "state.sqlite3", "bot")
        await queue.initialize()
        for post_id in (1, 2):
            await queue.enqueue([action(post_id)], post_id)
            prepared = await model._accept_action(
                await model._prepare_action(action(post_id))
            )
            if post_id == 2:
                await store.append_event(
                    prepared["observer"].run_id,
                    "run.generation_failed",
                    {"error": "model failed"},
                )
            await queue.status(post_id, "sent", 100 + post_id)
        await store.recover_forum_runs("bot")
        cid = (await store.list_conversations())[0].id
        assert [r["status"] for r in await store.list_forum_runs(cid)] == [
            "completed",
            "failed",
        ]

    asyncio.run(run())


def test_queue_order_is_stable_after_status_updates(tmp_path):
    async def run():
        queue = ForumQueue(tmp_path / "state.sqlite3", "bot")
        await queue.initialize()
        await queue.enqueue([action(1), action(2)], 2)
        await queue.status(1, "running")
        await queue.status(1, "pending")
        assert [post_id for post_id, _ in await queue.pending()] == [1, 2]

    asyncio.run(run())


def test_prechecks_are_bounded_and_do_not_create_early_records(tmp_path, monkeypatch):
    monkeypatch.setenv("SHUIYUAN_STATE_DIR", str(tmp_path))

    async def run():
        model, store = await setup(tmp_path)
        queue = ForumQueue(tmp_path / "state.sqlite3", "bot")
        await queue.initialize()
        await queue.enqueue([action(i, i) for i in range(1, 5)], 4)
        entered, release = asyncio.Event(), asyncio.Event()
        current = maximum = 0

        async def prepare(act):
            nonlocal current, maximum
            current += 1
            maximum = max(current, maximum)
            if current == 3:
                entered.set()
            await release.wait()
            current -= 1
            return None

        model._prepare_action = prepare
        worker = asyncio.create_task(model.watch_new_action_routine())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert current == maximum == 3
            assert (await store.forum_monitor())["runs"] == []
            assert await store.list_conversations() == []
        finally:
            release.set()
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await BaseUserActionModel.aclose(model)
        assert maximum == 3

    asyncio.run(run())


def test_snapshot_cursor_and_counts_share_one_read_view(tmp_path):
    async def run():
        model, store = await setup(tmp_path)
        prepared = await model._prepare_action(action())

        async def writer():
            for _ in range(12):
                await store.accept_forum_run(prepared["request"], "chat")

        task = asyncio.create_task(writer())
        while not task.done():
            snapshot = await store.forum_monitor()
            assert all(
                r["last_event_id"] <= snapshot["cursor"] for r in snapshot["runs"]
            )
            assert sum(c["queued_count"] for c in snapshot["conversations"]) == len(
                snapshot["runs"]
            )
        await task
        snapshot = await store.forum_monitor()
        assert len(snapshot["runs"]) == 12
        await store.accept_forum_run(prepared["request"], "chat")
        assert len(await store.forum_events_after(snapshot["cursor"])) == 1

    asyncio.run(run())

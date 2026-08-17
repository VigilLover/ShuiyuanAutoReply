import unittest
import os
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from shuiyuan_auto_reply.bootstrap.container import ApplicationContainer
from shuiyuan_auto_reply.bootstrap.settings import AppSettings
from shuiyuan_auto_reply.shuiyuan.user_action_model import BaseUserActionModel


class FakeWatcher(BaseUserActionModel):
    async def _new_action_routine(self, action) -> None:
        return None


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_background_tasks_are_cancelled_and_observed(self):
        watcher = FakeWatcher(SimpleNamespace(), "bot", [5, 7])
        finished = asyncio.Event()

        async def background():
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        task = asyncio.create_task(background())
        watcher._bg_tasks.add(task)
        await asyncio.sleep(0)
        await watcher.aclose()
        self.assertTrue(task.cancelled())
        self.assertTrue(finished.is_set())
        self.assertEqual(watcher._bg_tasks, set())

    async def test_failed_api_startup_closes_forum_session(self):
        forum = SimpleNamespace(close=AsyncMock())
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}), patch(
            "shuiyuan_auto_reply.bootstrap.container.ShuiyuanModel.create",
            new=AsyncMock(return_value=forum),
        ), patch(
            "shuiyuan_auto_reply.bootstrap.container.MentionProviderFactory.create_api",
            side_effect=RuntimeError("startup failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                await ApplicationContainer.for_api()
        forum.close.assert_awaited_once()

    async def test_container_closes_every_resource_exactly_once(self):
        managed = SimpleNamespace(aclose=AsyncMock())
        chat = SimpleNamespace(aclose=AsyncMock())
        forum = SimpleNamespace(close=AsyncMock())
        container = ApplicationContainer(
            AppSettings(), forum, SimpleNamespace(), chat, managed=[managed]
        )

        with patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_memory_manager",
            new_callable=AsyncMock,
        ) as memory_close, patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_record_manager",
            new_callable=AsyncMock,
        ) as record_close, patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_neo4j_manager",
            new_callable=AsyncMock,
        ) as neo4j_close, patch(
            "shuiyuan_auto_reply.bootstrap.container.close_shared_session",
            new_callable=AsyncMock,
        ) as image_close:
            await container.aclose()
            await container.aclose()

        managed.aclose.assert_awaited_once()
        chat.aclose.assert_awaited_once()
        forum.close.assert_awaited_once()
        memory_close.assert_awaited_once()
        record_close.assert_awaited_once()
        neo4j_close.assert_awaited_once()
        image_close.assert_awaited_once()

    async def test_one_close_failure_does_not_leak_other_resources(self):
        managed = SimpleNamespace(aclose=AsyncMock(side_effect=RuntimeError("boom")))
        chat = SimpleNamespace(aclose=AsyncMock())
        forum = SimpleNamespace(close=AsyncMock())
        container = ApplicationContainer(
            AppSettings(), forum, SimpleNamespace(), chat, managed=[managed]
        )
        with patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_memory_manager",
            new_callable=AsyncMock,
        ), patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_record_manager",
            new_callable=AsyncMock,
        ), patch(
            "shuiyuan_auto_reply.bootstrap.container.close_global_async_neo4j_manager",
            new_callable=AsyncMock,
        ), patch(
            "shuiyuan_auto_reply.bootstrap.container.close_shared_session",
            new_callable=AsyncMock,
        ):
            await container.aclose()
        chat.aclose.assert_awaited_once()
        forum.close.assert_awaited_once()

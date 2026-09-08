import asyncio
import os
import tempfile
import unittest
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
        with patch(
            "shuiyuan_auto_reply.bootstrap.deployment.get_deployment",
            return_value=SimpleNamespace(section=lambda _: {"shutdown_timeout": 0.01}),
        ):
            await watcher.aclose()
        self.assertTrue(task.cancelled())
        self.assertTrue(finished.is_set())
        self.assertEqual(watcher._bg_tasks, set())

    async def test_api_startup_defers_forum_and_chat_in_local_and_remote(self):
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
        from shuiyuan_auto_reply.infrastructure.forum.lazy import LazyChat, LazyForum

        config = get_deployment()
        for profile in ("local", "remote"):
            with (
                self.subTest(profile=profile),
                tempfile.TemporaryDirectory() as temp,
                patch.dict(
                    os.environ,
                    {
                        "SHUIYUAN_STATE_DIR": temp,
                        "MENTION_CHAT_PROVIDER": "deepseek",
                        "DEEPSEEK_API_KEY": "test-key",
                    },
                ),
                patch(
                    "shuiyuan_auto_reply.bootstrap.deployment.get_deployment",
                    return_value=SimpleNamespace(
                        profile=profile, section=config.section
                    ),
                ),
                patch(
                    "shuiyuan_auto_reply.bootstrap.container.ShuiyuanModel.create",
                    new=AsyncMock(side_effect=RuntimeError("login failed")),
                ) as login,
                patch(
                    "shuiyuan_auto_reply.bootstrap.container.MentionProviderFactory.create",
                    side_effect=RuntimeError("model unavailable"),
                ) as build,
            ):
                container = await ApplicationContainer.for_api()
                try:
                    self.assertIsInstance(container.forum_model, LazyForum)
                    self.assertIsInstance(
                        container.chat_handler._backend.model, LazyChat
                    )
                    login.assert_not_awaited()
                    build.assert_not_called()
                finally:
                    await container.aclose()
                login.assert_not_awaited()
                build.assert_not_called()

    async def test_failed_api_startup_closes_lazy_forum(self):
        forum = SimpleNamespace(close=AsyncMock())
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {"SHUIYUAN_STATE_DIR": temp}),
            patch(
                "shuiyuan_auto_reply.infrastructure.forum.lazy.LazyForum",
                return_value=forum,
            ),
            patch.object(
                ApplicationContainer,
                "_settings_for_profile",
                side_effect=RuntimeError("startup failed"),
            ),
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

        with (
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_memory_manager",
                new_callable=AsyncMock,
            ) as memory_close,
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_record_manager",
                new_callable=AsyncMock,
            ) as record_close,
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_neo4j_manager",
                new_callable=AsyncMock,
            ) as neo4j_close,
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_shared_session",
                new_callable=AsyncMock,
            ) as image_close,
        ):
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
        with (
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_memory_manager",
                new_callable=AsyncMock,
            ),
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_postgres_record_manager",
                new_callable=AsyncMock,
            ),
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_global_async_neo4j_manager",
                new_callable=AsyncMock,
            ),
            patch(
                "shuiyuan_auto_reply.bootstrap.container.close_shared_session",
                new_callable=AsyncMock,
            ),
        ):
            await container.aclose()
        chat.aclose.assert_awaited_once()
        forum.close.assert_awaited_once()

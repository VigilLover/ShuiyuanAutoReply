import asyncio
import base64
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from shuiyuan_auto_reply.application import BotContext, BotService, HandlerRegistry
from shuiyuan_auto_reply.bootstrap.container import ApplicationContainer
from shuiyuan_auto_reply.bootstrap.container import _SwappableBotService
from shuiyuan_auto_reply.bootstrap.settings import AppSettings, ProviderSettings
from shuiyuan_auto_reply.domain import (
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
    ReplyResult,
)
from shuiyuan_auto_reply.features.mention.image_generation import (
    ImageGenerationService,
)
from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.infrastructure.persistence import (
    LocalSecretVault,
    SQLiteExecutionObserver,
    SQLiteSessionRepository,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.interfaces.api.app import create_app


class HistoryChat:
    name = "chat"
    priority = 40

    async def matches(self, _context: BotContext) -> bool:
        return True

    async def handle(self, context: BotContext) -> ReplyResult:
        return ReplyResult(f"history={len(context.history)}:{context.request.content}")


def web_request(external_id: str, content: str = "hello") -> ReplyRequest:
    return ReplyRequest(
        request_id=f"web:{external_id}:{content}",
        conversation=ConversationRef(
            Channel.WEB, external_id, "wolf_lumine", "wolf_lumine"
        ),
        actor=ActorRef(Channel.WEB, external_id, "web-user"),
        content=content,
        dispatch_mode=DispatchMode.AUTO,
    )


class SQLiteStageTwoTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "state.sqlite3"
        self.store = SQLiteStateStore(self.path)
        await self.store.initialize()

    async def asyncTearDown(self):
        self.temp.cleanup()

    def service(self) -> BotService:
        return BotService(
            SQLiteSessionRepository(self.store),
            HandlerRegistry([HistoryChat()]),
            observer_factory=lambda: SQLiteExecutionObserver(self.store),
        )

    async def test_history_survives_store_recreation_and_epoch_clear(self):
        first = web_request("account-a", "one")
        await self.service().reply(first)

        reopened = SQLiteStateStore(self.path)
        await reopened.initialize()
        service = BotService(
            SQLiteSessionRepository(reopened), HandlerRegistry([HistoryChat()])
        )
        result = await service.reply(web_request("account-a", "two"))
        self.assertEqual(result.text, "history=2:two")

        await service.clear_conversation(first.conversation)
        self.assertEqual(
            await SQLiteSessionRepository(reopened).load(first.conversation), []
        )
        conversation = await reopened.ensure_conversation(first.conversation)
        records = await reopened.list_messages(conversation.id)
        self.assertEqual(records[-1].content, "上下文已清除")
        self.assertGreater(conversation.context_epoch + 1, 0)

    async def test_web_and_forum_namespaces_and_conversations_do_not_collide(self):
        web = web_request("42")
        forum_ref = ConversationRef(Channel.FORUM, "topic:42", "wolf_lumine", "wolf_lumine")
        self.assertEqual(web.actor.memory_id, "web:42")
        self.assertNotEqual(web.conversation, forum_ref)
        web_record = await self.store.ensure_conversation(web.conversation)
        forum_record = await self.store.ensure_conversation(forum_ref)
        self.assertNotEqual(web_record.id, forum_record.id)

    async def test_execution_payloads_are_redacted(self):
        request = web_request("redaction")
        conversation = await self.store.ensure_conversation(request.conversation)
        run_id = await self.store.create_run(request.request_id, conversation.id)
        await self.store.append_event(
            run_id,
            "tool.started",
            {
                "api_key": "top-secret",
                "arguments": "data:image/png;base64,QUJDRA==",
                "authorization": "Bearer abc.def",
            },
        )
        event = (await self.store.list_events_for_request(request.request_id))[0]
        encoded = str(event.payload)
        self.assertNotIn("top-secret", encoded)
        self.assertNotIn("QUJDRA", encoded)
        self.assertNotIn("abc.def", encoded)

    async def test_secret_vault_encrypts_values_and_only_exposes_metadata(self):
        vault = LocalSecretVault(self.store, Path(self.temp.name) / "master.key")
        await vault.set("web:openrouter", "sk-example-1234")
        self.assertEqual(await vault.get("web:openrouter"), "sk-example-1234")
        metadata = await vault.metadata("web:openrouter")
        self.assertEqual(metadata["last_four"], "1234")
        self.assertNotIn(b"sk-example-1234", self.path.read_bytes())
        self.assertEqual((Path(self.temp.name) / "master.key").stat().st_mode & 0o777, 0o600)

    async def test_web_profile_defaults_to_forum_deepseek_model_key_and_fallback(self):
        settings = AppSettings(
            providers=ProviderSettings(
                deepseek_api_key="same-key",
                deepseek_model="deepseek-primary",
                deepseek_fallback_model="deepseek-fallback",
            )
        )
        defaults = ApplicationContainer._profile_defaults(settings)
        self.assertEqual(defaults["provider"], "deepseek")
        self.assertEqual(defaults["model"], "deepseek-primary")
        self.assertEqual(defaults["fallback_model"], "deepseek-fallback")
        container = ApplicationContainer(
            settings,
            SimpleNamespace(),
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            state_store=self.store,
            secret_vault=LocalSecretVault(
                self.store, Path(self.temp.name) / "profile-master.key"
            ),
        )
        effective = await container._settings_for_profile("web", defaults)
        self.assertEqual(effective.deepseek_api_key, "same-key")

    async def test_web_handler_order_is_help_then_rua_then_chat(self):
        chat_model = SimpleNamespace(
            get_pumpkin_response=AsyncMock(return_value=("chat-body", ())),
            clear_session_history=lambda _key: None,
            aclose=AsyncMock(),
        )
        pet = SimpleNamespace(get_rua_response=AsyncMock(return_value="pet-body"))
        with patch(
            "shuiyuan_auto_reply.bootstrap.container.MentionPetModel",
            return_value=pet,
        ):
            handler, service = ApplicationContainer._build_web_service(
                chat_model, ProviderSettings(), self.store
            )
            help_result = await service.reply(
                web_request("handlers-help", "【帮助】【rua】")
            )
            self.assertIn("本地网页对话", help_result.text)
            pet.get_rua_response.assert_not_awaited()

            rua_result = await service.reply(web_request("handlers-rua", "【rua】轻轻"))
            self.assertEqual(rua_result.text, "pet-body")
            self.assertNotIn("data-signature", rua_result.text)

            chat_result = await service.reply(web_request("handlers-chat", "普通问题"))
            self.assertEqual(chat_result.text, "chat-body")
            call = chat_model.get_pumpkin_response.await_args
            self.assertIsNone(call.args[0])
            self.assertFalse(call.kwargs["load_forum_context"])
            await handler.aclose()


class ImageArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_managed_generation_saves_artifact_without_forum_upload(self):
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ,
            {
                "SHUIYUAN_STATE_DIR": temp,
                "IMAGE_GEN_API_KEY": "test-key",
                "IMAGE_GEN_API_URL": "https://images.example/v1",
            },
            clear=False,
        ):
            store = SQLiteStateStore(Path(temp) / "state.sqlite3")
            await store.initialize()
            forum = SimpleNamespace(upload_image=AsyncMock())
            service = ImageGenerationService(forum, store)
            with patch(
                "shuiyuan_auto_reply.features.mention.image_generation._request_image_bytes",
                new=AsyncMock(return_value=png),
            ):
                content, artifact = await service.generate(
                    "一幅足够详细的测试图片描述"
                )
            self.assertIn("artifact://", content)
            self.assertTrue(Path(artifact.local_path).is_file())
            forum.upload_image.assert_not_awaited()


class RuntimeSwitchTests(unittest.IsolatedAsyncioTestCase):
    async def test_old_runtime_closes_only_after_its_inflight_request_finishes(self):
        started = asyncio.Event()
        release = asyncio.Event()

        class OldService:
            async def reply(self, _request):
                started.set()
                await release.wait()
                return ReplyResult("old")

            async def clear_conversation(self, _conversation):
                return None

        class NewService:
            async def reply(self, _request):
                return ReplyResult("new")

            async def clear_conversation(self, _conversation):
                return None

        old_handler = SimpleNamespace(aclose=AsyncMock())
        new_handler = SimpleNamespace(aclose=AsyncMock())
        proxy = _SwappableBotService(OldService(), old_handler)
        running = asyncio.create_task(proxy.reply(web_request("runtime")))
        await started.wait()
        await proxy.swap(NewService(), new_handler)
        old_handler.aclose.assert_not_awaited()
        self.assertEqual((await proxy.reply(web_request("runtime-new"))).text, "new")
        release.set()
        self.assertEqual((await running).text, "old")
        old_handler.aclose.assert_awaited_once()
        await proxy.aclose()
        new_handler.aclose.assert_awaited_once()


class McpConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_mcp_tools_default_enabled_independently_of_builtin_allowlist(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.provider_settings = SimpleNamespace(mcp_server_url="http://mcp.test/sse")
        model.enabled_tools = {"builtin"}
        model.disabled_mcp_tools = set()
        model.state_store = None
        model.prompt_scope = SimpleNamespace(value="web")
        mcp_tool = SimpleNamespace(name="get_system_time")
        builtin_tool = SimpleNamespace(name="builtin")
        model._load_mcp_tools = AsyncMock(return_value=[mcp_tool])
        model._load_shuiyuan_tools = MagicMock(return_value=[builtin_tool])
        model.memory_model = SimpleNamespace(initialize=AsyncMock(), tools=[])
        model.openai_tools = []
        bound = SimpleNamespace(with_retry=MagicMock(return_value="bound"))
        model.llm = SimpleNamespace(bind_tools=MagicMock(return_value=bound))
        model._build_graph = MagicMock(return_value="graph")

        await model.initialize_agent()

        bound_tools = model.llm.bind_tools.call_args.args[0]
        self.assertEqual([tool.name for tool in bound_tools], ["get_system_time", "builtin"])

    async def test_disabled_mcp_tool_is_not_bound(self):
        model = MentionChatModel.__new__(MentionChatModel)
        model.provider_settings = SimpleNamespace(mcp_server_url="http://mcp.test/sse")
        model.enabled_tools = None
        model.disabled_mcp_tools = {"get_system_time"}
        model.state_store = None
        model.prompt_scope = SimpleNamespace(value="web")
        mcp_tool = SimpleNamespace(name="get_system_time")
        model._load_mcp_tools = AsyncMock(return_value=[mcp_tool])
        model._load_shuiyuan_tools = MagicMock(return_value=[])
        model.memory_model = SimpleNamespace(initialize=AsyncMock(), tools=[])
        model.openai_tools = []
        bound = SimpleNamespace(with_retry=MagicMock(return_value="bound"))
        model.llm = SimpleNamespace(bind_tools=MagicMock(return_value=bound))
        model._build_graph = MagicMock(return_value="graph")

        await model.initialize_agent()

        self.assertEqual(model.llm.bind_tools.call_args.args[0], [])


class ManagedApiTests(unittest.TestCase):
    def test_login_web_chat_clear_and_forum_read_only(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {"SHUIYUAN_STATE_DIR": temp}, clear=False
        ):
            store = SQLiteStateStore(Path(temp) / "state.sqlite3")
            asyncio.run(store.initialize())

            app_settings = AppSettings(
                providers=ProviderSettings(
                    mcp_server_url="http://localhost:58000/sse"
                )
            )

            class Container:
                settings = app_settings
                chat_handler = SimpleNamespace(_backend=SimpleNamespace(model=None))
                state_store = store
                secret_vault = LocalSecretVault(store, Path(temp) / "master.key")
                bot_service = BotService(
                    SQLiteSessionRepository(store), HandlerRegistry([HistoryChat()])
                )

                async def aclose(self):
                    return None

            async def factory():
                return Container()

            app = create_app(factory)
            with TestClient(app) as client:
                token = client.app.state.admin_auth.token
                self.assertEqual(
                    client.post("/api/admin/login", json={"token": "wrong"}).status_code,
                    403,
                )
                login = client.post("/api/admin/login", json={"token": token})
                self.assertEqual(login.status_code, 200)
                profiles = client.get("/api/settings/profiles").json()
                web_profile = next(p for p in profiles if p["scope"] == "web")
                self.assertEqual(web_profile["active"]["provider"], "deepseek")
                self.assertEqual(web_profile["active"]["disabled_mcp_tools"], [])
                discovered = [
                    SimpleNamespace(name="get_system_time", description="查询时间")
                ]
                with patch(
                    "shuiyuan_auto_reply.interfaces.api.app.MentionChatModel._load_mcp_tools",
                    new=AsyncMock(return_value=discovered),
                ):
                    mcp = client.get("/api/settings/mcp/web").json()
                self.assertTrue(mcp["connected"])
                self.assertEqual(mcp["url"], "http://localhost:58000/sse")
                self.assertEqual(mcp["tools"][0]["name"], "get_system_time")
                self.assertTrue(mcp["tools"][0]["enabled"])
                self.assertEqual(
                    client.post("/api/conversations", json={}).status_code, 403
                )
                client.headers["X-CSRF-Token"] = login.json()["csrf_token"]
                created = client.post("/api/conversations", json={}).json()
                response = client.post(
                    f"/api/conversations/{created['id']}/messages/stream",
                    json={"message": "hello"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("message.completed", response.text)
                detail = client.get(f"/api/conversations/{created['id']}").json()
                self.assertEqual([m["role"] for m in detail["messages"]], ["user", "assistant"])
                client.post(f"/api/conversations/{created['id']}/clear")
                detail = client.get(f"/api/conversations/{created['id']}").json()
                self.assertEqual(detail["messages"][-1]["content"], "上下文已清除")

                forum_ref = ConversationRef(
                    Channel.FORUM, "topic:9", "wolf_lumine", "wolf_lumine"
                )
                forum_record = asyncio.run(store.ensure_conversation(forum_ref))
                forbidden = client.post(
                    f"/api/conversations/{forum_record.id}/messages/stream",
                    json={"message": "must not publish"},
                )
                self.assertEqual(forbidden.status_code, 403)

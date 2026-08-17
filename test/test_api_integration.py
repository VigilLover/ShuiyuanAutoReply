import unittest

from fastapi.testclient import TestClient

from shuiyuan_auto_reply.application import BotContext, BotService, HandlerRegistry
from shuiyuan_auto_reply.bootstrap.settings import AppSettings
from shuiyuan_auto_reply.domain import ReplyResult
from shuiyuan_auto_reply.infrastructure.persistence import InMemorySessionRepository
from shuiyuan_auto_reply.interfaces.api.app import create_app


class HistoryAwareChat:
    name = "chat"
    priority = 40

    async def matches(self, context: BotContext) -> bool:
        return True

    async def handle(self, context: BotContext) -> ReplyResult:
        return ReplyResult(f"history={len(context.history)}:{context.request.content}")


class FakeContainer:
    def __init__(self):
        self.settings = AppSettings()
        self.bot_service = BotService(
            InMemorySessionRepository(), HandlerRegistry([HistoryAwareChat()])
        )
        self.close_count = 0

    async def aclose(self):
        self.close_count += 1


class ApiIntegrationTests(unittest.TestCase):
    def test_chat_clear_session_isolation_and_lifespan(self):
        container = FakeContainer()

        async def factory():
            return container

        app = create_app(factory)
        with TestClient(app) as client:
            first_a = client.post(
                "/api/chat",
                json={"session_id": "a", "token": "ta", "message": "one"},
            )
            second_a = client.post(
                "/api/chat",
                json={"session_id": "a", "token": "ta", "message": "two"},
            )
            first_b = client.post(
                "/api/chat",
                json={"session_id": "b", "token": "tb", "message": "one"},
            )
            self.assertEqual(first_a.json()["reply"], "history=0:one")
            self.assertEqual(second_a.json()["reply"], "history=2:two")
            self.assertEqual(first_b.json()["reply"], "history=0:one")

            cleared = client.post(
                "/api/clear", json={"session_id": "a", "token": "ta"}
            )
            self.assertEqual(cleared.status_code, 200)
            after_clear = client.post(
                "/api/chat",
                json={"session_id": "a", "token": "ta", "message": "again"},
            )
            continued_b = client.post(
                "/api/chat",
                json={"session_id": "b", "token": "tb", "message": "two"},
            )
            self.assertEqual(after_clear.json()["reply"], "history=0:again")
            self.assertEqual(continued_b.json()["reply"], "history=2:two")
            self.assertEqual(client.get("/api/health").json()["active_sessions_count"], 2)

        self.assertEqual(container.close_count, 1)

    def test_http_contract_validation_and_token_status_codes(self):
        container = FakeContainer()

        async def factory():
            return container

        with TestClient(create_app(factory)) as client:
            empty = client.post(
                "/api/chat",
                json={"session_id": "a", "token": "t", "message": "  "},
            )
            self.assertEqual(empty.status_code, 400)
            self.assertEqual(empty.json()["detail"], "消息不能为空")

            client.post(
                "/api/chat",
                json={"session_id": "a", "token": "t", "message": "hello"},
            )
            forbidden = client.post(
                "/api/chat",
                json={"session_id": "a", "token": "wrong", "message": "hello"},
            )
            self.assertEqual(forbidden.status_code, 403)

            clear_forbidden = client.post(
                "/api/clear", json={"session_id": "a", "token": "wrong"}
            )
            self.assertEqual(clear_forbidden.status_code, 403)

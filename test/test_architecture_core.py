import unittest

from shuiyuan_auto_reply.application import BotContext, BotService, HandlerRegistry
from shuiyuan_auto_reply.domain import (
    ActorRef,
    Channel,
    ConversationRef,
    DispatchMode,
    ReplyRequest,
    ReplyResult,
)
from shuiyuan_auto_reply.infrastructure.persistence import InMemorySessionRepository


def request(session: str, mode: DispatchMode = DispatchMode.CHAT_ONLY) -> ReplyRequest:
    return ReplyRequest(
        request_id=f"request:{session}",
        conversation=ConversationRef(Channel.API, session, "bot", "persona"),
        actor=ActorRef(Channel.API, session, "NULL"),
        content=f"message:{session}",
        dispatch_mode=mode,
    )


class FakeHandler:
    def __init__(self, name: str, priority: int, matches: bool = True):
        self.name = name
        self.priority = priority
        self.should_match = matches
        self.calls = 0

    async def matches(self, context: BotContext) -> bool:
        return self.should_match

    async def handle(self, context: BotContext) -> ReplyResult:
        self.calls += 1
        return ReplyResult(f"{self.name}:{context.request.content}")


class ConversationIdentityTests(unittest.TestCase):
    def test_forum_handler_order_is_frozen(self):
        from shuiyuan_auto_reply.features.mention.mention_model import (
            MENTION_HANDLER_PRIORITIES,
        )

        self.assertEqual(
            list(MENTION_HANDLER_PRIORITIES),
            ["help", "rua", "clear", "chat", "dice", "poll"],
        )

    def test_forum_and_api_keys_cannot_collide(self):
        forum = ConversationRef(Channel.FORUM, "topic:123", "bot", "persona")
        api = ConversationRef(Channel.API, "123", "bot", "persona")
        self.assertNotEqual(forum, api)
        self.assertEqual(forum.session_id, 123)
        self.assertEqual(api.session_id, "api:123")

    def test_api_actor_memory_scope_is_prefixed(self):
        actor = ActorRef(Channel.API, "123", "NULL")
        self.assertEqual(actor.memory_id, "api:123")
        self.assertNotEqual(actor.memory_id, "123")


class BotServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sessions = InMemorySessionRepository()
        self.chat = FakeHandler("chat", 40)
        self.service = BotService(
            self.sessions, HandlerRegistry([self.chat]), chat_handler_name="chat"
        )

    async def test_two_api_sessions_are_isolated(self):
        first = request("a")
        second = request("b")
        await self.service.reply(first)
        await self.service.reply(second)
        first_history = await self.sessions.load(first.conversation)
        second_history = await self.sessions.load(second.conversation)
        self.assertEqual([m.content for m in first_history], ["message:a", "chat:message:a"])
        self.assertEqual([m.content for m in second_history], ["message:b", "chat:message:b"])

    async def test_clear_a_does_not_clear_b(self):
        first = request("a")
        second = request("b")
        await self.service.reply(first)
        await self.service.reply(second)
        await self.service.clear_conversation(first.conversation)
        self.assertEqual(await self.sessions.load(first.conversation), [])
        self.assertEqual(len(await self.sessions.load(second.conversation)), 2)

    async def test_chat_only_calls_explicit_chat_handler(self):
        earlier = FakeHandler("help", 10, matches=True)
        chat = FakeHandler("chat", 40, matches=False)
        service = BotService(
            InMemorySessionRepository(), HandlerRegistry([chat, earlier])
        )
        result = await service.reply(request("chat-only"))
        self.assertEqual(result.text, "chat:message:chat-only")
        self.assertEqual(earlier.calls, 0)

    async def test_auto_uses_priority_order(self):
        help_handler = FakeHandler("help", 10, matches=True)
        chat = FakeHandler("chat", 40, matches=True)
        service = BotService(
            InMemorySessionRepository(), HandlerRegistry([chat, help_handler])
        )
        result = await service.reply(request("auto", DispatchMode.AUTO))
        self.assertEqual(result.text, "help:message:auto")
        self.assertEqual(chat.calls, 0)

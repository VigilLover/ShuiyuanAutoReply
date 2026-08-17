"""Explicit chat handler; CHAT_ONLY never relies on an implicit fallback."""

from typing import Protocol

from shuiyuan_auto_reply.application.dispatch import BotContext
from shuiyuan_auto_reply.domain import ConversationRef, ReplyRequest, ReplyResult
from .callback import CallbackMessageHandler


class ChatBackend(Protocol):
    async def reply(self, request: ReplyRequest) -> ReplyResult: ...
    async def clear(self, conversation: ConversationRef) -> None: ...
    async def aclose(self) -> None: ...


class ChatHandler:
    name = "chat"
    priority = 40

    def __init__(self, backend: ChatBackend) -> None:
        self._backend = backend

    async def matches(self, context: BotContext) -> bool:
        return True

    async def handle(self, context: BotContext) -> ReplyResult:
        return await self._backend.reply(context.request)

    async def clear_conversation(self, conversation: ConversationRef) -> None:
        await self._backend.clear(conversation)

    async def aclose(self) -> None:
        await self._backend.aclose()


class CallbackChatHandler(CallbackMessageHandler):
    def __init__(self, predicate, callback, priority: int = 40) -> None:
        super().__init__(
            name="chat", priority=priority, predicate=predicate, callback=callback
        )

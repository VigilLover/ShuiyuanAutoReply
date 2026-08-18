"""Process-local session storage used by the first architecture phase."""

import asyncio

from shuiyuan_auto_reply.application.ports.session import SessionRepository
from shuiyuan_auto_reply.domain import (
    ChatMessage,
    ConversationRef,
    ReplyRequest,
    ReplyResult,
)


class InMemorySessionRepository(SessionRepository):
    def __init__(self) -> None:
        self._messages: dict[ConversationRef, list[ChatMessage]] = {}
        self._lock = asyncio.Lock()

    async def load(self, key: ConversationRef) -> list[ChatMessage]:
        async with self._lock:
            return list(self._messages.get(key, ()))

    async def append(
        self, key: ConversationRef, request: ReplyRequest, result: ReplyResult
    ) -> None:
        async with self._lock:
            history = self._messages.setdefault(key, [])
            history.extend(
                (ChatMessage("user", request.content), ChatMessage("assistant", result.text))
            )

    async def clear(self, key: ConversationRef) -> None:
        async with self._lock:
            self._messages.pop(key, None)

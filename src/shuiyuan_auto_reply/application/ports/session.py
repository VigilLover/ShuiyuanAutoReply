from typing import Protocol

from shuiyuan_auto_reply.domain import (
    ChatMessage,
    ConversationRef,
    ReplyRequest,
    ReplyResult,
)


class SessionRepository(Protocol):
    async def load(self, key: ConversationRef) -> list[ChatMessage]: ...

    async def append(
        self, key: ConversationRef, request: ReplyRequest, result: ReplyResult
    ) -> None: ...

    async def clear(self, key: ConversationRef) -> None: ...

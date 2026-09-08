"""Small callback adapter used while legacy feature logic is migrated."""

from collections.abc import Awaitable, Callable

from shuiyuan_auto_reply.application.dispatch import BotContext
from shuiyuan_auto_reply.domain import ReplyResult


class CallbackMessageHandler:
    def __init__(
        self,
        *,
        name: str,
        priority: int,
        predicate: Callable[[str], bool],
        callback: Callable[[BotContext], Awaitable[str | ReplyResult | None]],
    ) -> None:
        self.name = name
        self.priority = priority
        self._predicate = predicate
        self._callback = callback

    async def matches(self, context: BotContext) -> bool:
        return self._predicate(context.request.content)

    async def handle(self, context: BotContext) -> ReplyResult:
        value = await self._callback(context)
        if value is None:
            raise LookupError(f"Handler {self.name!r} matched but returned no result")
        return value if isinstance(value, ReplyResult) else ReplyResult(value)

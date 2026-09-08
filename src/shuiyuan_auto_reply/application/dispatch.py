"""Explicit, priority-based message dispatch."""

from dataclasses import dataclass
from typing import Protocol, Sequence

from shuiyuan_auto_reply.domain import ChatMessage, ReplyRequest, ReplyResult


@dataclass(frozen=True, slots=True)
class BotContext:
    request: ReplyRequest
    history: tuple[ChatMessage, ...]


class MessageHandler(Protocol):
    name: str
    priority: int

    async def matches(self, context: BotContext) -> bool: ...

    async def handle(self, context: BotContext) -> ReplyResult: ...


class HandlerRegistry:
    """Immutable ordered view of registered handlers."""

    def __init__(self, handlers: Sequence[MessageHandler]) -> None:
        self._handlers = tuple(sorted(handlers, key=lambda handler: handler.priority))
        names = [handler.name for handler in self._handlers]
        if len(names) != len(set(names)):
            raise ValueError("Handler names must be unique")

    @property
    def handlers(self) -> tuple[MessageHandler, ...]:
        return self._handlers

    def by_name(self, name: str) -> MessageHandler:
        for handler in self._handlers:
            if handler.name == name:
                return handler
        raise LookupError(f"Handler {name!r} is not registered")

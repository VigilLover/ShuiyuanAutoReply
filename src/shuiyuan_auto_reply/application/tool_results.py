"""Request-local tool evidence. No model, transport, or persistence dependencies."""

import asyncio
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4

PAGE_CHARS = 12_000


@dataclass
class TurnResults:
    results: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, asyncio.Task] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    cache_hits: int = 0

    def save(self, value: Any) -> str:
        key = str(uuid4())
        self.results[key] = value
        return key

    def read(self, result_id: str, cursor: int = 0) -> dict:
        if result_id not in self.results:
            return {"status": "error", "error": "unknown_result", "retryable": False}
        value = self.results[result_id]
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        if cursor < 0 or cursor > len(text):
            return {"status": "error", "error": "invalid_cursor", "retryable": False}
        end = min(cursor + PAGE_CHARS, len(text))
        return {
            "result_id": result_id,
            "content": text[cursor:end],
            "total_chars": len(text),
            "truncated": end < len(text),
            "next_cursor": end if end < len(text) else None,
        }

    async def query(
        self, key: str, call: Callable[[], Awaitable[Any]], *, refresh: bool = False
    ) -> Any:
        if not refresh and key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if key in self.pending:
            self.cache_hits += 1
            return await self.pending[key]
        task = asyncio.create_task(call())
        self.pending[key] = task
        try:
            result = await task
            if (
                result is not None
                and not isinstance(result, str)
                and not (
                    isinstance(result, dict)
                    and result.get("status") in {"error", "partial"}
                )
            ):
                self.cache[key] = result
            return result
        finally:
            self.pending.pop(key, None)


current_turn: ContextVar[TurnResults | None] = ContextVar("tool_results", default=None)


async def cached_query(
    key: str, call: Callable[[], Awaitable[Any]], *, refresh: bool = False
) -> Any:
    turn = current_turn.get()
    return await turn.query(key, call, refresh=refresh) if turn else await call()


def tool_error(exc: Exception) -> dict:
    return {
        "status": "error",
        "error": type(exc).__name__,
        "message": str(exc)[:300],
        "retryable": isinstance(exc, (TimeoutError, ConnectionError)),
    }

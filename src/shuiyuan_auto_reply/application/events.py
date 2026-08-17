"""Request-scoped structured execution events without coupling to persistence."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Protocol

from shuiyuan_auto_reply.domain import ReplyRequest, ReplyResult


class ExecutionObserver(Protocol):
    async def start(self, request: ReplyRequest) -> str | None: ...
    async def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None: ...
    async def finish(self, result: ReplyResult | None, error: Exception | None = None) -> None: ...


_observer: ContextVar[ExecutionObserver | None] = ContextVar("execution_observer", default=None)
_run_id: ContextVar[str | None] = ContextVar("execution_run_id", default=None)
_memory_scope: ContextVar[str | None] = ContextVar("memory_scope", default=None)


def set_execution_context(
    observer: ExecutionObserver | None,
    run_id: str | None,
    memory_scope: str | None = None,
):
    return (
        _observer.set(observer),
        _run_id.set(run_id),
        _memory_scope.set(memory_scope),
    )


def reset_execution_context(tokens) -> None:
    observer_token, run_token, memory_token = tokens
    _observer.reset(observer_token)
    _run_id.reset(run_token)
    _memory_scope.reset(memory_token)


def current_run_id() -> str | None:
    return _run_id.get()


def current_memory_scope() -> str | None:
    return _memory_scope.get()


async def emit_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
    observer = _observer.get()
    if observer is not None:
        await observer.emit(event_type, payload)

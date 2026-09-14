"""Request-local tool evidence. No model, transport, or persistence dependencies."""

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .retrieval_control import RetrievalControl
from .task_progress import TaskProgress, content_digest, source_records

PAGE_CHARS = 12_000


@dataclass
class TurnResults:
    control: RetrievalControl = field(default_factory=RetrievalControl)
    progress: TaskProgress = field(default_factory=TaskProgress)
    evidence: dict[str, dict] = field(default_factory=dict)
    execution_ids: set[str] = field(default_factory=set)
    read_pages: set[tuple] = field(default_factory=set)
    external_requests: int = 0
    image_failures: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    index_ids: set[str] = field(default_factory=set)
    cache: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, asyncio.Task] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    cache_hits: int = 0
    deadline: float = field(default_factory=lambda: time.monotonic() + 900)

    def save(self, value: Any, *, index: bool = False) -> str:
        text = (
            value
            if isinstance(value, str)
            else json.dumps(
                value,
                ensure_ascii=False,
                default=lambda obj: (
                    obj.to_dict() if hasattr(obj, "to_dict") else str(obj)
                ),
            )
        )
        digest = sha256(text.encode()).hexdigest()
        key = self.digests.get(digest)
        if key is None:
            key = str(uuid4())
            self.digests[digest] = key
            self.results[key] = text
        if index:
            self.index_ids.add(key)
        return key

    def observe(self, value: Any, *, tool: str = "") -> set[str]:
        added = set()
        for identity, record in source_records(value):
            kind = "full" if tool in {"get_post", "get_post_by_id"} else "summary"
            digest = content_digest(record)
            key = identity + ":" + kind + ":" + digest[:16]
            if key in self.evidence:
                continue
            result_id = self.save(record)
            author = record.get("author") or {}
            self.evidence[key] = {
                "identity": identity,
                "kind": kind,
                "result_id": result_id,
                "username": author.get("username", record.get("username")),
                "topic_id": record.get("topic_id"),
                "post_number": record.get("post_number"),
                "reply_to_post_number": record.get("reply_to_post_number"),
                "preview": str(record.get("content", record.get("snippet", record)))[
                    :300
                ],
            }
            if (
                not self.progress.authors
                or not self.evidence[key]["username"]
                or self.evidence[key]["username"].casefold()
                in {a.casefold() for a in self.progress.authors}
            ):
                added.add(key)
        return added

    def read(
        self,
        result_id: str,
        cursor: int = 0,
        *,
        field: str | None = None,
        limit: int = PAGE_CHARS,
    ) -> dict:
        result_id = self.evidence.get(result_id, {}).get("result_id", result_id)
        if result_id not in self.results:
            return {"status": "error", "error": "unknown_result", "retryable": False}
        value = self.results[result_id]
        if field is not None:
            try:
                value = json.loads(value)[field]
            except (ValueError, KeyError, TypeError):
                return {"status": "error", "error": "unknown_field", "retryable": False}
        if not 1 <= limit <= PAGE_CHARS:
            return {"status": "error", "error": "invalid_limit", "retryable": False}
        self.read_pages.add((result_id, field, cursor))
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        if cursor < 0 or cursor > len(text):
            return {"status": "error", "error": "invalid_cursor", "retryable": False}
        end = min(cursor + limit, len(text))
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
        if refresh:
            self.cache.pop(key, None)
        self.external_requests += 1
        task = asyncio.create_task(call())
        self.pending[key] = task
        try:
            result = await task
            if (
                result is not None
                and not isinstance(result, str)
                and not getattr(result, "warnings", None)
                and not (hasattr(result, "raw") and result.raw is None)
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
        "error": getattr(exc, "code", type(exc).__name__),
        "status_code": getattr(exc, "status_code", None),
        "message": str(exc)[:300],
        "retryable": getattr(
            exc, "retryable", isinstance(exc, (TimeoutError, ConnectionError))
        ),
    }


def turn_scope(func):
    """Give every invocation an isolated store, including nested/concurrent turns."""
    from functools import wraps

    @wraps(func)
    async def wrapped(*args, **kwargs):
        turn = TurnResults()
        token = current_turn.set(turn)
        try:
            return await func(*args, **kwargs)
        finally:
            for task in turn.pending.values():
                task.cancel()
            if turn.pending:
                await asyncio.gather(*turn.pending.values(), return_exceptions=True)
            current_turn.reset(token)

    return wrapped

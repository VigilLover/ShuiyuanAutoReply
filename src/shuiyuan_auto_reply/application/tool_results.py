"""Request-local tool evidence. No model, transport, or persistence dependencies."""

import asyncio
import json
import re
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
    forum_http_requests: int = 0
    started_at: float = field(default_factory=time.monotonic)
    media_bytes: dict[str, bytes] = field(default_factory=dict)
    media_digests: dict[str, int] = field(default_factory=dict)
    image_failures: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    index_ids: set[str] = field(default_factory=set)
    cache: dict[str, Any] = field(default_factory=dict)
    pending: dict[str, asyncio.Task] = field(default_factory=dict)
    references: dict[str, Any] = field(default_factory=dict)
    cursors: dict[str, Any] = field(default_factory=dict)
    topic_coverage: dict[int, set[int]] = field(default_factory=dict)
    completed_topics: set[int] = field(default_factory=set)
    topic_titles: dict[int, str] = field(default_factory=dict)
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
            kind = (
                "full"
                if tool in {"forum_read", "web_read", "get_chuangka_menu"}
                else "summary"
            )
            existing = self.evidence.get(identity)
            if existing and (existing["kind"] == "full" or kind == "summary"):
                continue
            result_id = self.save(record)
            author = record.get("author") or {}
            author_username = (
                author.get("username") if isinstance(author, dict) else str(author)
            )
            self.evidence[identity] = {
                "identity": identity,
                "kind": kind,
                "digest": content_digest(record),
                "result_id": result_id,
                "username": author_username or record.get("username"),
                "topic_id": record.get("topic_id"),
                "post_number": record.get("post_number"),
                "reply_to_post_number": record.get("reply_to_post_number"),
                "preview": str(record.get("content", record.get("snippet", record)))[
                    :300
                ],
            }
            added.add(identity)
        return added

    def note_topic_page(
        self, topic_id: int, items: list[dict], *, complete: bool, title: str = ""
    ) -> None:
        coverage = self.topic_coverage.setdefault(topic_id, set())
        for item in items:
            ref = str(item.get("ref", ""))
            match = re.fullmatch(rf"forum:{topic_id}/(\d+)", ref)
            if match:
                coverage.add(int(match.group(1)))
        if title:
            self.topic_titles[topic_id] = title
        if complete:
            self.completed_topics.add(topic_id)

    @staticmethod
    def _topic_only_search(args: dict) -> int | None:
        if any(args.get(key) for key in ("username", "after_date", "before_date")):
            return None
        topic_id = args.get("topic_id")
        query = str(args.get("query", "") or "").strip()
        matches = re.findall(r"(?<!\S)topic:(\d+)(?=\s|$)", query, re.I)
        if topic_id is None and matches:
            topic_id = int(matches[0])
        query = re.sub(r"(?<!\S)topic:\d+(?=\s|$)", "", query, flags=re.I)
        query = re.sub(r"(?<!\S)order:(?:latest|oldest)(?=\s|$)", "", query, flags=re.I)
        return int(topic_id) if topic_id and not query.strip() else None

    def is_redundant_completed_call(self, tool: str, args: dict) -> bool:
        if tool not in {"forum_search", "forum_read"}:
            return False
        request = args
        if args.get("cursor"):
            request = self.cursors.get(args["cursor"], {}).get("request", {})
        if tool == "forum_read":
            if request.get("post_id") or request.get("post_number"):
                return False
            topic_id = request.get("topic_id")
        else:
            topic_id = self._topic_only_search(request)
        return bool(topic_id and int(topic_id) in self.completed_topics)

    def final_evidence_text(self, max_chars: int) -> str:
        """Render every canonical source into a compact, evenly sized digest."""
        records = []
        for evidence in self.evidence.values():
            raw = self.results.get(evidence["result_id"], "")
            try:
                record = json.loads(raw)
            except (TypeError, ValueError):
                record = {"text": str(raw)}
            if not isinstance(record, dict):
                record = {"text": str(record)}
            item = {
                key: record[key]
                for key in (
                    "ref",
                    "title",
                    "topic",
                    "author",
                    "username",
                    "created_at",
                    "reply_to",
                    "text",
                    "content",
                )
                if record.get(key) not in (None, "", [], {})
            }
            item.setdefault("ref", evidence["identity"])
            records.append(item)
        if not records:
            return "[]"
        allowance = max(120, max_chars // len(records) - 80)
        while allowance >= 20:
            compact = []
            for record in records:
                item = dict(record)
                for field in ("text", "content"):
                    value = item.get(field)
                    if isinstance(value, str) and len(value) > allowance:
                        item[field] = value[: allowance - 1].rstrip() + "…"
                compact.append(item)
            text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
            if len(text) <= max_chars:
                return text
            allowance = int(allowance * 0.7)
        references = [record["ref"] for record in records]
        return json.dumps(
            {"refs": references, "detail_truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
        )

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
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str)
        )
        if cursor < 0 or cursor > len(text):
            return {"status": "error", "error": "invalid_cursor", "retryable": False}
        end = min(cursor + limit, len(text))
        if result_id not in self.index_ids and end > cursor:
            self.read_pages.add((sha256(text[cursor:end].encode()).hexdigest(),))
        return {
            "result_id": result_id,
            "content": text[cursor:end],
            "total_chars": len(text),
            "page_limit": limit,
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
        if not key.startswith("image:"):
            self.external_requests += 1
        task = asyncio.create_task(call())
        self.pending[key] = task
        try:
            result = await task
            if (
                isinstance(result, dict)
                and result.get("status") == "error"
                and result.get("retryable") is False
            ):
                self.cache[key] = result
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

"""SQLite implementation of the structured execution observer."""

from __future__ import annotations

from typing import Any

from shuiyuan_auto_reply.domain import ReplyRequest, ReplyResult

from .state import SQLiteStateStore


class SQLiteExecutionObserver:
    def __init__(self, store: SQLiteStateStore, *, provider: str | None = None, model: str | None = None) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.run_id: str | None = None
        self.conversation_id: str | None = None
        self.usage: dict[str, int] = {}

    async def start(self, request: ReplyRequest) -> str:
        conversation = await self.store.ensure_conversation(request.conversation)
        self.conversation_id = conversation.id
        self.run_id = await self.store.create_run(request.request_id, conversation.id, provider=self.provider, model=self.model)
        await self.store.append_event(self.run_id, "run.started", {"channel": request.conversation.channel.value})
        return self.run_id

    async def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if event_type == "usage.recorded" and payload:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = payload.get(key)
                if isinstance(value, int):
                    self.usage[key] = value
        if self.run_id:
            await self.store.append_event(self.run_id, event_type, payload)

    async def finish(self, result: ReplyResult | None, error: Exception | None = None) -> None:
        if not self.run_id:
            return
        if error is not None:
            await self.store.append_event(self.run_id, "run.failed", {"error": str(error)[:1000]})
            await self.store.finish_run(self.run_id, status="failed", error=str(error)[:2000])
            if self.conversation_id:
                await self.store.append_message(self.conversation_id, "system", f"请求失败：{str(error)[:500]}", run_id=self.run_id, status="failed")
            return
        await self.store.append_event(self.run_id, "run.completed", {"attachments": len(result.attachments) if result else 0})
        await self.store.finish_run(self.run_id, status="completed", usage=self.usage)

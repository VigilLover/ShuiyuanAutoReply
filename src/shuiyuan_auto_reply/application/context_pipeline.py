"""Stable context-pipeline boundary for the current LangGraph implementation."""

from typing import Any, Protocol

from shuiyuan_auto_reply.domain import ReplyRequest


class ContextPipeline(Protocol):
    async def load(self, request: ReplyRequest) -> dict[str, Any]: ...

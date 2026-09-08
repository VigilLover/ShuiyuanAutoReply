from typing import Any, Protocol

from shuiyuan_auto_reply.domain.capabilities import ProviderCapabilities


class LLMProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...
    async def invoke(self, messages: Any, tools: Any) -> Any: ...

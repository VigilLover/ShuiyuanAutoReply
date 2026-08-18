from typing import Any, Protocol

from shuiyuan_auto_reply.domain import ReplyRequest


class ChannelContextProvider(Protocol):
    async def load(self, request: ReplyRequest) -> Any: ...

"""Forum and non-forum context boundaries."""

from dataclasses import dataclass

from shuiyuan_auto_reply.domain import ReplyRequest


@dataclass(frozen=True, slots=True)
class ChannelContext:
    recent_messages: str = "无近期回帖记录"


class EmptyChannelContextProvider:
    async def load(self, request: ReplyRequest) -> ChannelContext:
        return ChannelContext()


class ForumChannelContextProvider:
    def __init__(self, loader) -> None:
        self._loader = loader

    async def load(self, request: ReplyRequest) -> ChannelContext:
        if request.forum_context is None:
            raise ValueError("forum_context is required")
        return ChannelContext(
            await self._loader(request.forum_context.topic_id)
        )

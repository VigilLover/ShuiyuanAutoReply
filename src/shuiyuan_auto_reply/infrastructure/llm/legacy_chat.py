"""Adapter keeping the proven LangGraph/model pipeline behind the new port."""

from shuiyuan_auto_reply.domain import Channel, ConversationRef, ReplyRequest, ReplyResult
from shuiyuan_auto_reply.shuiyuan.objects import User


class LegacyMentionChatBackend:
    def __init__(self, model) -> None:
        self.model = model

    async def reply(self, request: ReplyRequest) -> ReplyResult:
        forum = request.forum_context
        user_id: int | str
        if request.actor.channel is Channel.FORUM and request.actor.external_id.isdigit():
            user_id = int(request.actor.external_id)
        else:
            user_id = request.actor.memory_id
        user = User(
            id=user_id,
            username=request.actor.username,
            name=request.actor.display_name,
        )
        text = await self.model.get_pumpkin_response(
            forum.topic_id if forum else None,
            forum.reply_to_post_number if forum else None,
            request.content,
            user,
            session_id=request.conversation.session_id,
            load_forum_context=forum is not None,
            memory_user_id=request.actor.memory_id if forum is None else user_id,
        )
        return ReplyResult(text=text or "")

    async def clear(self, conversation: ConversationRef) -> None:
        self.model.clear_session_history(conversation.session_id)

    async def aclose(self) -> None:
        await self.model.aclose()

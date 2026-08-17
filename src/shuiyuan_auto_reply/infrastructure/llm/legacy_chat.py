"""Adapter keeping the proven LangGraph/model pipeline behind the new port."""

from shuiyuan_auto_reply.domain import AttachmentRef, Channel, ChatMessage, ConversationRef, ReplyRequest, ReplyResult
from shuiyuan_auto_reply.shuiyuan.objects import User


class LegacyMentionChatBackend:
    def __init__(self, model, *, owns_model: bool = True) -> None:
        self.model = model
        self.owns_model = owns_model

    async def reply(
        self, request: ReplyRequest, history: tuple[ChatMessage, ...] = ()
    ) -> ReplyResult:
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
        text, artifacts = await self.model.get_pumpkin_response(
            forum.topic_id if forum else None,
            forum.reply_to_post_number if forum else None,
            request.content,
            user,
            session_id=request.conversation.session_id,
            load_forum_context=forum is not None,
            memory_user_id=request.actor.memory_id if forum is None else user_id,
            external_history=history,
            include_artifacts=True,
        )
        return ReplyResult(
            text=text or "",
            attachments=tuple(
                AttachmentRef(artifact.uri, artifact.mime_type, artifact.artifact_id)
                for artifact in artifacts
            ),
        )

    async def clear(self, conversation: ConversationRef) -> None:
        self.model.clear_session_history(conversation.session_id)

    async def aclose(self) -> None:
        if self.owns_model:
            await self.model.aclose()

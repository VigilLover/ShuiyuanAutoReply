"""Inbound request and history message types."""

from dataclasses import dataclass
from enum import Enum

from .conversation import ActorRef, ConversationRef, ForumContextRef


class DispatchMode(str, Enum):
    AUTO = "auto"
    CHAT_ONLY = "chat_only"


@dataclass(frozen=True, slots=True)
class AttachmentRef:
    url: str
    media_type: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ReplyRequest:
    request_id: str
    conversation: ConversationRef
    actor: ActorRef
    content: str
    dispatch_mode: DispatchMode
    forum_context: ForumContextRef | None = None
    attachments: tuple[AttachmentRef, ...] = ()

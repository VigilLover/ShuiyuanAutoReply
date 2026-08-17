"""Framework-free domain model for Shuiyuan Auto Reply."""

from .conversation import ActorRef, Channel, ConversationRef, ForumContextRef
from .message import AttachmentRef, ChatMessage, DispatchMode, ReplyRequest
from .response import ReplyResult

__all__ = [
    "ActorRef",
    "AttachmentRef",
    "Channel",
    "ChatMessage",
    "ConversationRef",
    "DispatchMode",
    "ForumContextRef",
    "ReplyRequest",
    "ReplyResult",
]

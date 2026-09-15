"""Framework-free domain model for Shuiyuan Auto Reply."""

from .conversation import ActorRef, Channel, ConversationRef, ForumContextRef
from .message import AttachmentRef, ChatMessage, DispatchMode, ReplyRequest
from .response import (
    ForumMediaRef,
    GeneratedImageArtifact,
    ReplyGenerationError,
    ReplyResult,
    VisualMediaArtifact,
)

__all__ = [
    "ActorRef",
    "AttachmentRef",
    "Channel",
    "ChatMessage",
    "ConversationRef",
    "DispatchMode",
    "ForumContextRef",
    "ForumMediaRef",
    "GeneratedImageArtifact",
    "ReplyGenerationError",
    "ReplyRequest",
    "ReplyResult",
    "VisualMediaArtifact",
]

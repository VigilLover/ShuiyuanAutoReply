"""Channel-independent bot result."""

from dataclasses import dataclass

from .message import AttachmentRef


@dataclass(frozen=True, slots=True)
class ReplyResult:
    text: str
    attachments: tuple[AttachmentRef, ...] = ()

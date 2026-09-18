"""Channel-independent bot result."""

from dataclasses import dataclass

from .message import AttachmentRef

UNKNOWN_REPLY_TEXT = "抱歉，发生了未知错误"


@dataclass(frozen=True, slots=True)
class GeneratedImageArtifact:
    artifact_id: str
    mime_type: str
    local_path: str
    byte_count: int
    width: int | None = None
    height: int | None = None

    @property
    def uri(self) -> str:
        return f"artifact://{self.artifact_id}"


@dataclass(frozen=True, slots=True)
class VisualMediaArtifact:
    artifact_id: str
    mime_type: str
    local_path: str
    byte_count: int
    source_kind: str
    source_url: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None

    @property
    def uri(self) -> str:
        return f"artifact://{self.artifact_id}"


@dataclass(frozen=True, slots=True)
class ForumMediaRef:
    short_path: str


@dataclass(frozen=True, slots=True)
class ReplyResult:
    text: str
    attachments: tuple[AttachmentRef, ...] = ()
    input_attachments: tuple[AttachmentRef, ...] = ()


class ReplyGenerationError(Exception):
    """Generation failed after a user-facing fallback text was already chosen.

    Raising this instead of swallowing the failure keeps the run record honest:
    callers mark the run failed while still publishing ``fallback_text`` verbatim.
    """

    def __init__(self, message: str, *, fallback_text: str = "") -> None:
        super().__init__(message)
        self.fallback_text = fallback_text

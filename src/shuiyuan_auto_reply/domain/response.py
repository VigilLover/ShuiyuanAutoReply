"""Channel-independent bot result."""

from dataclasses import dataclass

from .message import AttachmentRef


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
class ForumMediaRef:
    short_path: str


@dataclass(frozen=True, slots=True)
class ReplyResult:
    text: str
    attachments: tuple[AttachmentRef, ...] = ()

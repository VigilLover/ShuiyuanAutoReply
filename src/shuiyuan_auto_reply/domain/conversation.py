"""Conversation identities shared by all inbound channels."""

from dataclasses import dataclass
from enum import Enum


class Channel(str, Enum):
    FORUM = "forum"
    API = "api"
    WEB = "web"


@dataclass(frozen=True, slots=True)
class ActorRef:
    channel: Channel
    external_id: str
    username: str
    display_name: str | None = None

    @property
    def memory_id(self) -> str:
        """Return an ID whose namespace cannot collide across channels."""
        if self.channel in {Channel.API, Channel.WEB}:
            return f"{self.channel.value}:{self.external_id}"
        return self.external_id


@dataclass(frozen=True, slots=True)
class ConversationRef:
    channel: Channel
    external_id: str
    bot_id: str
    persona_id: str

    @property
    def session_id(self) -> str | int:
        """Bridge key used by the first, in-memory legacy session adapter.

        Forum topics intentionally retain their numeric key. Other channels are
        explicitly prefixed so an API session named ``123`` cannot share history
        with forum topic 123.
        """
        if self.channel is Channel.FORUM and self.external_id.startswith("topic:"):
            value = self.external_id.removeprefix("topic:")
            return int(value) if value.isdigit() else self.external_id
        return f"{self.channel.value}:{self.external_id}"


@dataclass(frozen=True, slots=True)
class ForumContextRef:
    topic_id: int
    post_id: int | None = None
    post_number: int | None = None
    reply_to_post_number: int | None = None

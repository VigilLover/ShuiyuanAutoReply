"""Request-local image handles so the model never copies real image addresses.

The model only sees short tokens: ``[图N]`` for images produced or selected in
the current turn (displayable in the reply) and ``#hN`` for images that appeared
in earlier replies (usable only as generation references).  The harness keeps
the real artifact or URL behind each token and resolves it after generation.
"""

import re
from dataclasses import dataclass, field
from typing import Any

DISPLAY_TOKEN_RE = re.compile(r"图\s*(\d{1,2})")
HISTORY_TOKEN_RE = re.compile(r"#h(\d{1,3})\b")


def display_token(index: int) -> str:
    return f"[图{index}]"


def history_token(index: int) -> str:
    return f"#h{index}"


@dataclass
class DisplayImage:
    artifact: Any
    description: str = ""


@dataclass
class HistoryImage:
    url: str
    description: str = ""


@dataclass
class TurnImageRegistry:
    display: list[DisplayImage] = field(default_factory=list)
    history: list[HistoryImage] = field(default_factory=list)

    def register(self, artifact: Any, description: str = "") -> str:
        """Return the ``[图N]`` token for an artifact, reusing an existing one."""
        artifact_id = getattr(artifact, "artifact_id", None)
        for index, item in enumerate(self.display, 1):
            if getattr(item.artifact, "artifact_id", None) == artifact_id:
                return display_token(index)
        self.display.append(DisplayImage(artifact, description))
        return display_token(len(self.display))

    def register_history(self, url: str, description: str = "") -> str:
        for index, item in enumerate(self.history, 1):
            if item.url == url:
                return history_token(index)
        self.history.append(HistoryImage(url, description))
        return history_token(len(self.history))

    def resolve_display(self, index: int) -> DisplayImage | None:
        if 1 <= index <= len(self.display):
            return self.display[index - 1]
        return None

    def resolve_reference(self, token: str) -> Any | None:
        """Map a reference token to an artifact (``[图N]``) or a URL (``#hN``).

        Returns ``None`` for anything that is not a handle, so callers can fall
        back to treating the value as a plain URL.
        """
        value = str(token).strip()
        match = HISTORY_TOKEN_RE.fullmatch(value)
        if match:
            index = int(match.group(1))
            if 1 <= index <= len(self.history):
                return self.history[index - 1].url
            raise KeyError(value)
        match = DISPLAY_TOKEN_RE.fullmatch(value.strip("[]【】"))
        if match:
            item = self.resolve_display(int(match.group(1)))
            if item is None:
                raise KeyError(value)
            return item.artifact
        return None

    def display_summary(self) -> str:
        return "、".join(
            display_token(index)
            + (f"（{item.description}）" if item.description else "")
            for index, item in enumerate(self.display, 1)
        )

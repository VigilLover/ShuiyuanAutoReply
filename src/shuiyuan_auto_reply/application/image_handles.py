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
_BRACKETED_DISPLAY_RE = re.compile(r"[\[【]\s*图\s*(\d{1,2})\s*[\]】]")


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

    def expand_handles(self, text: str) -> str:
        """Swap handles back to real addresses in text that outlives the turn."""

        def history(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if 1 <= index <= len(self.history):
                return self.history[index - 1].url
            return match.group(0)

        def display(match: re.Match[str]) -> str:
            item = self.resolve_display(int(match.group(1)))
            return item.artifact.uri if item else match.group(0)

        return _BRACKETED_DISPLAY_RE.sub(display, HISTORY_TOKEN_RE.sub(history, text))

    def display_summary(self) -> str:
        return "、".join(
            display_token(index)
            + (f"（{item.description}）" if item.description else "")
            for index, item in enumerate(self.display, 1)
        )


_TOKEN = r"[\[【]?\s*图\s*(?P<n>\d{1,2})\s*[\]】]?"
_MARKDOWN_TOKEN_RE = re.compile(r"!\[(?P<alt>[^\]\n]*)\]\(\s*" + _TOKEN + r"\s*\)")
_BARE_TOKEN_RE = re.compile(r"[\[【]\s*图\s*(?P<n>\d{1,2})\s*[\]】](?!\()")
_ANY_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.I)
_RAW_IMAGE_ADDRESS_RE = re.compile(
    r"(?:upload|artifact)://[^\s)\]>\"'\\]+|/api/artifacts/[\w-]+"
)
# Handles are turn-local: a copied history label or old handle must not survive.
_HISTORY_LABEL_RE = re.compile(r"[\[【](?:历史图[^\]】\n]*|旧图)[\]】]|#h\d{1,3}\b")
_STALE_HANDLE_RE = re.compile(r"[\[【]\s*图\s*\d{1,2}\s*[\]】]|#h\d{1,3}\b")
_PLACEHOLDER = "\x00IMG{}\x00"


@dataclass
class RenderedImages:
    text: str
    used: list[Any]
    rejected: list[str]


def render_image_placeholders(text: str, registry: TurnImageRegistry) -> RenderedImages:
    """Turn ``[图N]`` into ``![alt](artifact://…)`` and drop every other image.

    Only images registered in this turn can be shown.  Image Markdown, HTML
    images, and raw upload/artifact addresses written by the model are removed,
    so a hallucinated or history-copied link can never reach the forum.
    """
    used: list[Any] = []
    rejected: list[str] = []
    rendered: list[str] = []

    def keep(index: int, alt: str) -> str | None:
        item = registry.resolve_display(index)
        if item is None:
            return None
        if item.artifact not in used:
            used.append(item.artifact)
        alt = alt.strip() or item.description or "图片"
        rendered.append(f"![{alt}]({item.artifact.uri})")
        return _PLACEHOLDER.format(len(rendered) - 1)

    def markdown_token(match: re.Match[str]) -> str:
        result = keep(int(match.group("n")), match.group("alt"))
        if result is None:
            rejected.append(match.group(0))
            return ""
        return result

    def bare_token(match: re.Match[str]) -> str:
        result = keep(int(match.group("n")), "")
        if result is None:
            rejected.append(match.group(0))
            return ""
        return result

    def reject(match: re.Match[str]) -> str:
        rejected.append(match.group(0))
        return ""

    text = _MARKDOWN_TOKEN_RE.sub(markdown_token, text)
    text = _BARE_TOKEN_RE.sub(bare_token, text)
    text = _ANY_MARKDOWN_IMAGE_RE.sub(reject, text)
    text = _HTML_IMAGE_RE.sub(reject, text)
    text = _RAW_IMAGE_ADDRESS_RE.sub(reject, text)
    text = _HISTORY_LABEL_RE.sub(reject, text)
    for index, markdown in enumerate(rendered):
        text = text.replace(_PLACEHOLDER.format(index), markdown)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return RenderedImages(text.strip(), used, rejected)


_HISTORY_MARKDOWN_RE = re.compile(r"!\[(?P<alt>[^\]\n]*)\]\((?P<url>[^)\s]+)[^)\n]*\)")
_HISTORY_HTML_RE = re.compile(
    r"<img\b[^>]*?\bsrc=[\"'](?P<url>[^\"']+)[\"'][^>]*>", re.I
)


def strip_history_images(text: str, registry: TurnImageRegistry) -> str:
    """Replace image addresses in earlier messages with ``#hN`` reference handles.

    Handles already present in stored history (for example in a saved tool-call
    summary) belong to an earlier turn and would point at a different image
    now, so they are neutralized before this turn's handles are assigned.
    """

    def label(url: str, alt: str) -> str:
        # Discourse writes size hints into the alt text: ``image|690x388``.
        alt = alt.split("|", 1)[0].strip()
        if alt.lower() in {"image", "图片"}:
            alt = ""
        handle = registry.register_history(url, alt)
        return f"[历史图 {handle}" + (f"：{alt}]" if alt else "]")

    text = _STALE_HANDLE_RE.sub("[旧图]", text)
    text = _HISTORY_MARKDOWN_RE.sub(
        lambda m: label(m.group("url"), m.group("alt")), text
    )
    text = _HISTORY_HTML_RE.sub(lambda m: label(m.group("url"), ""), text)
    return _RAW_IMAGE_ADDRESS_RE.sub(
        lambda m: registry.register_history(m.group(0)), text
    )


def remove_images(text: str) -> str:
    """Drop images from text that only serves as a tone reference."""
    text = _HISTORY_MARKDOWN_RE.sub("", text)
    text = _HISTORY_HTML_RE.sub("", text)
    return _RAW_IMAGE_ADDRESS_RE.sub("", text)

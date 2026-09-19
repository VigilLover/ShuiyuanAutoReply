"""Central environment mapping with lazy subsystem validation."""

import os
from dataclasses import dataclass, field
from enum import Enum

from .deployment import get_deployment


def _value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _text(name: str, default: str) -> str:
    value = _value(name, default)
    return default if value is None else value


def _flag(name: str, default: bool = False) -> bool:
    value = _value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ForumSettings:
    cookie_file: str = field(
        default_factory=lambda: get_deployment().section("forum")["cookie_file"]
    )
    bot_username: str = field(
        default_factory=lambda: get_deployment().section("forum")["bot_username"]
    )


class DeepSeekApiFormat(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """DeepSeek is the only chat provider; embeddings are configured elsewhere."""

    mention_provider: str = "deepseek"
    # Empty means DeepSeek's own endpoint; the settings UI can point a stored
    # configuration at any OpenAI-compatible base URL.
    mention_base_url: str | None = field(
        default_factory=lambda: _value("MENTION_BASE_URL")
    )
    deepseek_api_key: str | None = field(
        default_factory=lambda: _value("DEEPSEEK_API_KEY")
    )
    deepseek_model: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_MODEL", "deepseek-flash")
    )
    deepseek_api_format: DeepSeekApiFormat = field(
        default_factory=lambda: DeepSeekApiFormat(
            _text(
                "DEEPSEEK_MENTION_API_FORMAT",
                DeepSeekApiFormat.CHAT_COMPLETIONS.value,
            )
            .strip()
            .lower()
        )
    )
    deepseek_thinking: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_THINKING", "enabled")
        .strip()
        .lower()
    )
    # Effort for investigation rounds (tool planning) and for the final answer.
    deepseek_reasoning_effort: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_REASONING_EFFORT", "high")
        .strip()
        .lower()
    )
    deepseek_final_reasoning_effort: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_FINAL_REASONING_EFFORT", "high")
        .strip()
        .lower()
    )
    _deepseek_max_tokens: str | None = field(
        default_factory=lambda: _value("DEEPSEEK_MENTION_MAX_TOKENS"), repr=False
    )
    _deepseek_request_timeout: str | None = field(
        default_factory=lambda: _value("DEEPSEEK_MENTION_REQUEST_TIMEOUT"),
        repr=False,
    )
    pet_model: str = field(
        default_factory=lambda: _text("PET_REPLY_MODEL", "deepseek-v4-pro")
    )
    mcp_server_url: str | None = field(default_factory=lambda: _value("MCP_SERVER_URL"))

    def validate_forum(self) -> None:
        if self.mention_provider != "deepseek":
            raise ValueError(
                f"MENTION_CHAT_PROVIDER must be deepseek; got {self.mention_provider!r}."
            )
        if not self.deepseek_api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

    def validate_deepseek_options(self) -> None:
        try:
            DeepSeekApiFormat(self.deepseek_api_format)
        except ValueError as exc:
            raise ValueError(
                "DEEPSEEK_MENTION_API_FORMAT must be chat_completions or responses"
            ) from exc
        if self.deepseek_thinking not in {"enabled", "disabled"}:
            raise ValueError("DEEPSEEK_MENTION_THINKING must be enabled or disabled")
        for name, value in (
            ("DEEPSEEK_MENTION_REASONING_EFFORT", self.deepseek_reasoning_effort),
            (
                "DEEPSEEK_MENTION_FINAL_REASONING_EFFORT",
                self.deepseek_final_reasoning_effort,
            ),
        ):
            if value not in {"low", "high", "max"}:
                raise ValueError(f"{name} must be low, high or max")

    @staticmethod
    def _parse_optional_positive(name: str, raw: str | None) -> int | None:
        if raw is None or not raw.strip():
            return None
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer; got {raw!r}.") from exc
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}.")
        return value

    @property
    def deepseek_max_tokens(self) -> int | None:
        return self._parse_optional_positive(
            "DEEPSEEK_MENTION_MAX_TOKENS", self._deepseek_max_tokens
        )

    @property
    def deepseek_request_timeout(self) -> int:
        """Hard cap for one model request in seconds; the turn deadline still applies."""
        return (
            self._parse_optional_positive(
                "DEEPSEEK_MENTION_REQUEST_TIMEOUT", self._deepseek_request_timeout
            )
            or 180
        )


@dataclass(frozen=True, slots=True)
class MemorySettings:
    search_limit: int = field(
        default_factory=lambda: int(_text("LANGMEM_SEARCH_LIMIT", "5"))
    )
    max_context_chars: int = field(
        default_factory=lambda: int(_text("LANGMEM_CONTEXT_MAX_CHARS", "1600"))
    )
    strict: bool = field(
        default_factory=lambda: _flag("POSTGRES_MEMORY_STRICT")
        or _flag("POSTGRES_STRICT")
    )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    style_top_k: int = 8


@dataclass(frozen=True, slots=True)
class ImageSettings:
    model: str | None = field(default_factory=lambda: _value("IMAGE_GEN_MODEL"))


@dataclass(frozen=True, slots=True)
class ApiSettings:
    host: str = "127.0.0.1"
    port: int = 11451


@dataclass(frozen=True, slots=True)
class AppSettings:
    forum: ForumSettings = field(default_factory=ForumSettings)
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    image: ImageSettings = field(default_factory=ImageSettings)
    api: ApiSettings = field(default_factory=ApiSettings)

"""Central environment mapping with lazy subsystem validation."""

import os
from dataclasses import dataclass, field


def _value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _text(name: str, default: str) -> str:
    value = _value(name, default)
    return default if value is None else value


def _cascading_text(primary: str, fallback: str, default: str) -> str:
    value = _value(primary)
    if value is not None:
        return value
    return _text(fallback, default)


def _flag(name: str, default: bool = False) -> bool:
    value = _value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ForumSettings:
    cookie_file: str = "cookies"
    bot_username: str = "wolf_lumine"


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    mention_provider: str = field(
        default_factory=lambda: _text("MENTION_CHAT_PROVIDER", "deepseek").strip().lower()
    )
    deepseek_api_key: str | None = field(default_factory=lambda: _value("DEEPSEEK_API_KEY"))
    dashscope_api_key: str | None = field(default_factory=lambda: _value("DASHSCOPE_API_KEY"))
    openrouter_api_key: str | None = field(default_factory=lambda: _value("OPENROUTER_API_KEY"))
    mimo_api_key: str | None = field(default_factory=lambda: _value("MIMO_API_KEY"))
    deepseek_model: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_MODEL", "deepseek-v4-pro")
    )
    deepseek_fallback_model: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_FALLBACK_MODEL", "deepseek-v4-flash")
    )
    deepseek_thinking: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_THINKING", "enabled").strip().lower()
    )
    deepseek_reasoning_effort: str = field(
        default_factory=lambda: _text("DEEPSEEK_MENTION_REASONING_EFFORT", "max").strip().lower()
    )
    _deepseek_max_tokens: str | None = field(
        default_factory=lambda: _value("DEEPSEEK_MENTION_MAX_TOKENS"), repr=False
    )
    dashscope_model: str = field(
        default_factory=lambda: _text("DASHSCOPE_MENTION_MODEL", "qwen3.5-plus-2026-02-15")
    )
    dashscope_fallback_model: str = field(
        default_factory=lambda: _text("DASHSCOPE_MENTION_FALLBACK_MODEL", "qwen3.5-plus")
    )
    openrouter_mention_model: str = field(
        default_factory=lambda: _cascading_text(
            "OPENROUTER_MENTION_MODEL",
            "OPENROUTER_MODEL",
            "google/gemini-3.1-flash-lite-preview",
        )
    )
    openrouter_proxy: str | None = field(default_factory=lambda: _value("OPENROUTER_PROXY"))
    mimo_model: str = field(
        default_factory=lambda: _text("MIMO_MENTION_MODEL", "mimo-v2.5")
    )
    mimo_thinking: str = field(
        default_factory=lambda: _text("MIMO_MENTION_THINKING", "enabled").strip().lower()
    )
    _mimo_max_tokens: str | None = field(
        default_factory=lambda: _value("MIMO_MENTION_MAX_TOKENS"), repr=False
    )
    _mimo_max_retries: str | None = field(
        default_factory=lambda: _value("MIMO_MENTION_MAX_RETRIES"), repr=False
    )
    _mimo_multimodal_search_images: str | None = field(
        default_factory=lambda: _value("MIMO_MULTIMODAL_MAX_SEARCH_IMAGES"),
        repr=False,
    )
    pet_model: str = field(
        default_factory=lambda: _text("PET_REPLY_MODEL", "deepseek-v4-pro")
    )
    mcp_server_url: str | None = field(default_factory=lambda: _value("MCP_SERVER_URL"))

    def validate_forum(self) -> None:
        allowed = {"deepseek", "tongyi", "openrouter", "mimo"}
        if self.mention_provider not in allowed:
            raise ValueError(
                "MENTION_CHAT_PROVIDER must be one of "
                f"{', '.join(sorted(allowed))}; got {self.mention_provider!r}."
            )
        keys = {
            "deepseek": self.deepseek_api_key,
            "tongyi": self.dashscope_api_key,
            "openrouter": self.openrouter_api_key,
            "mimo": self.mimo_api_key,
        }
        if not keys[self.mention_provider]:
            env_name = {
                "deepseek": "DEEPSEEK_API_KEY",
                "tongyi": "DASHSCOPE_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "mimo": "MIMO_API_KEY",
            }[self.mention_provider]
            raise ValueError(f"Please set the {env_name} environment variable.")

    def validate_api(self) -> None:
        if not self.openrouter_api_key:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

    def validate_deepseek_options(self) -> None:
        if self.deepseek_thinking not in {"enabled", "disabled"}:
            raise ValueError("DEEPSEEK_MENTION_THINKING must be enabled or disabled")
        if self.deepseek_reasoning_effort not in {"high", "max"}:
            raise ValueError("DEEPSEEK_MENTION_REASONING_EFFORT must be high or max")

    def validate_mimo_options(self) -> None:
        if self.mimo_thinking not in {"enabled", "disabled"}:
            raise ValueError("MIMO_MENTION_THINKING must be enabled or disabled")

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
    def mimo_max_tokens(self) -> int | None:
        return self._parse_optional_positive(
            "MIMO_MENTION_MAX_TOKENS", self._mimo_max_tokens
        )

    @property
    def mimo_max_retries(self) -> int:
        return self._parse_optional_positive(
            "MIMO_MENTION_MAX_RETRIES", self._mimo_max_retries
        ) or 3

    @property
    def mimo_multimodal_search_images(self) -> int:
        return self._parse_optional_positive(
            "MIMO_MULTIMODAL_MAX_SEARCH_IMAGES",
            self._mimo_multimodal_search_images,
        ) or 2


@dataclass(frozen=True, slots=True)
class MemorySettings:
    search_limit: int = field(default_factory=lambda: int(_text("LANGMEM_SEARCH_LIMIT", "5")))
    max_context_chars: int = field(default_factory=lambda: int(_text("LANGMEM_CONTEXT_MAX_CHARS", "1600")))
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
    host: str = "0.0.0.0"
    port: int = 11451


@dataclass(frozen=True, slots=True)
class AppSettings:
    forum: ForumSettings = field(default_factory=ForumSettings)
    providers: ProviderSettings = field(default_factory=ProviderSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    image: ImageSettings = field(default_factory=ImageSettings)
    api: ApiSettings = field(default_factory=ApiSettings)

import os
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import FallbackLLM, MentionChatModel

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DEEPSEEK_FALLBACK_MODEL = "deepseek-v4-flash"
DEEPSEEK_DEFAULT_MAX_RETRIES = 3
DEEPSEEK_DEFAULT_THINKING = "enabled"
DEEPSEEK_DEFAULT_REASONING_EFFORT = "max"


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that preserves DeepSeek thinking metadata.

    DeepSeek requires assistant ``reasoning_content`` to be sent back after
    thinking-mode tool calls. LangChain's generic OpenAI adapter currently drops
    this provider-specific field, so keep it in ``AIMessage.additional_kwargs``
    and re-inject it into later chat-completion payloads.
    """

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")

        payload_messages = payload.get("messages", [])
        for source_message, payload_message in zip(
            messages, payload_messages, strict=False
        ):
            if not isinstance(source_message, AIMessage):
                continue
            reasoning_content = source_message.additional_kwargs.get("reasoning_content")
            if reasoning_content and "reasoning_content" not in payload_message:
                payload_message["reasoning_content"] = reasoning_content

        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        response_dict = response if isinstance(response, dict) else response.model_dump()
        reasoning_by_index = [
            (choice.get("message") or {}).get("reasoning_content")
            for choice in response_dict.get("choices", [])
        ]

        result = super()._create_chat_result(response, generation_info)
        for generation, reasoning_content in zip(
            result.generations, reasoning_by_index, strict=False
        ):
            if reasoning_content:
                generation.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )
        return result


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(allowed))}; got {value!r}."
        )
    return value


def _env_optional_int(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


def _mk_deepseek_llm(api_key: str, model_name: str) -> ChatOpenAI:
    thinking = _env_choice(
        "DEEPSEEK_MENTION_THINKING",
        DEEPSEEK_DEFAULT_THINKING,
        {"enabled", "disabled"},
    )
    reasoning_effort = _env_choice(
        "DEEPSEEK_MENTION_REASONING_EFFORT",
        DEEPSEEK_DEFAULT_REASONING_EFFORT,
        {"high", "max"},
    )
    max_tokens = _env_optional_int("DEEPSEEK_MENTION_MAX_TOKENS")

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": DEEPSEEK_BASE_URL,
        "max_retries": DEEPSEEK_DEFAULT_MAX_RETRIES,
        "extra_body": {"thinking": {"type": thinking}},
    }
    if thinking == "enabled":
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return DeepSeekChatOpenAI(**kwargs)


class MentionDeepSeekModel(MentionChatModel):
    """
    A model for managing DeepSeek-backed conversation data.
    Falls back to deepseek-v4-flash if deepseek-v4-pro is unavailable.
    Both models use thinking mode with reasoning_effort=max.
    """

    def __init__(self, model: ShuiyuanModel, username: str = "wolf_lumine"):
        super().__init__(model, username=username)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

        model_name = os.getenv("DEEPSEEK_MENTION_MODEL", DEEPSEEK_DEFAULT_MODEL)
        fallback_name = os.getenv(
            "DEEPSEEK_MENTION_FALLBACK_MODEL", DEEPSEEK_FALLBACK_MODEL
        )

        self.llm = FallbackLLM(
            _mk_deepseek_llm(api_key, model_name),
            _mk_deepseek_llm(api_key, fallback_name),
        )

    def parse_model_output(self, raw_output: Any) -> str:
        """
        Parse the raw output from the model to extract the final response text.

        :param raw_output: The raw output from the model.
        :return: The extracted response text.
        """
        if raw_output is None:
            return ""
        if isinstance(raw_output, str):
            return raw_output.strip()

        res = ""
        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict) and "text" in item:
                    res += item["text"]
                if hasattr(item, "text"):
                    res += item.text
                if isinstance(item, str):
                    res += item
        return res.strip()

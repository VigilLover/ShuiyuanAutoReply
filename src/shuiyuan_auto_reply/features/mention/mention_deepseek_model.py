from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
from shuiyuan_auto_reply.application.ports.prompt import PromptScope

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


def _mk_deepseek_llm(
    api_key: str,
    model_name: str,
    provider_settings: ProviderSettings | None = None,
) -> ChatOpenAI:
    current = provider_settings or ProviderSettings()
    current.validate_deepseek_options()
    thinking = current.deepseek_thinking
    reasoning_effort = current.deepseek_reasoning_effort
    max_tokens = current.deepseek_max_tokens

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

    def __init__(
        self,
        model: ShuiyuanModel,
        username: str = "wolf_lumine",
        provider_settings: ProviderSettings | None = None,
        prompt_scope: PromptScope = PromptScope.FORUM,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
    ):
        super().__init__(model, username=username, prompt_scope=prompt_scope, enabled_tools=enabled_tools, disabled_mcp_tools=disabled_mcp_tools, state_store=state_store, system_prompt_override=system_prompt_override)

        current = provider_settings or ProviderSettings()
        api_key = current.deepseek_api_key
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

        model_name = current.deepseek_model
        fallback_name = current.deepseek_fallback_model

        self.llm = FallbackLLM(
            _mk_deepseek_llm(api_key, model_name, current),
            _mk_deepseek_llm(api_key, fallback_name, current),
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

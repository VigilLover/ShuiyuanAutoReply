from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import MentionChatModel

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5"
MIMO_DEFAULT_THINKING = "enabled"
MIMO_DEFAULT_MAX_TOKENS: int | None = None
MIMO_DEFAULT_MAX_RETRIES = 3


class MiMoChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant for Xiaomi MiMo's OpenAI-compatible endpoint."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if "max_tokens" in payload:
            payload["max_completion_tokens"] = payload.pop("max_tokens")

        payload_messages = payload.get("messages", [])
        for source_message, payload_message in zip(
            messages, payload_messages, strict=False
        ):
            if not isinstance(source_message, AIMessage):
                continue
            reasoning_content = source_message.additional_kwargs.get(
                "reasoning_content"
            )
            if reasoning_content and "reasoning_content" not in payload_message:
                payload_message["reasoning_content"] = reasoning_content

        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
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


def _mk_mimo_llm(
    api_key: str,
    model_name: str,
    provider_settings: ProviderSettings | None = None,
) -> ChatOpenAI:
    current = provider_settings or ProviderSettings()
    current.validate_mimo_options()
    thinking = current.mimo_thinking
    max_tokens = current.mimo_max_tokens
    max_retries = current.mimo_max_retries

    kwargs: dict[str, Any] = dict(
        model=model_name,
        api_key=api_key,
        base_url=MIMO_BASE_URL,
        default_headers={"api-key": api_key},
        max_retries=max_retries,
        extra_body={"thinking": {"type": thinking}},
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return MiMoChatOpenAI(**kwargs)


class MentionMimoModel(MentionChatModel):
    """Mention model backed by Xiaomi MiMo v2.5 with multimodal inputs."""

    def _get_multimodal_prompt_rules(self) -> str:
        return (
            "【图片理解 - 严格规则】\n"
            "1. 用户当前附件和 forum_read 精确读取结果中的图片会直接进入视觉输入。\n"
            "2. forum_search 和话题列表只返回图片引用；需要理解图片时，用 forum_read 精确定位帖子。\n"
            "3. 需要理解头像时调用 users，并设置 include_avatar=True；返回的头像会按用户标签进入视觉输入。\n"
            "4. 外部图片使用 web_read；无法载入时明确说明，不能根据 URL 猜测内容。\n\n"
        )

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
        super().__init__(
            model,
            username=username,
            prompt_scope=prompt_scope,
            enabled_tools=enabled_tools,
            disabled_mcp_tools=disabled_mcp_tools,
            state_store=state_store,
            system_prompt_override=system_prompt_override,
        )

        current = provider_settings or ProviderSettings()
        api_key = current.mimo_api_key
        if not api_key:
            raise ValueError("Please set the MIMO_API_KEY environment variable.")

        model_name = current.mimo_model
        self.llm = _mk_mimo_llm(api_key, model_name, current)
        self.supports_multimodal = True
        self.multimodal_search_image_limit = current.mimo_multimodal_search_images

    def parse_model_output(self, raw_output: Any) -> str:
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

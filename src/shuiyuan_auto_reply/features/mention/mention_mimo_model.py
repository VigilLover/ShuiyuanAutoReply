from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings

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
            "1. 如果用户要求查看、分析、理解某张图片、帖子搜索结果里的图片或用户头像，请先通过水源工具拿到图片 URL，再调用 inspect_image。\n"
            "2. search_posts、recent_posts、search_posts_by_time、get_post 的结果若包含 Images 字段，只说明帖子里有这些图片；不要直接猜测图片内容，需要看图时必须调用 inspect_image(image_url=...)。\n"
            "3. 需要理解用户头像时，先调用 search_user 或 search_user_by_id，并把 include_avatar 设为 True；拿到 avatar 后再调用 inspect_image，**必须传入 description 参数标明该头像对应的用户名或 ID（如 description='用户 xxx 的头像'）**。\n"
            "4. inspect_image 只接受水源图片或头像 URL。外部网页图片无法通过该工具读取。\n"
            "5. 当一次需要对多名用户头像调用 inspect_image 时，为每次调用分别传入不同的 description 以区分归属，避免模型混淆。\n\n"
        )

    def __init__(
        self,
        model: ShuiyuanModel,
        username: str = "wolf_lumine",
        provider_settings: ProviderSettings | None = None,
    ):
        super().__init__(model, username=username)

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

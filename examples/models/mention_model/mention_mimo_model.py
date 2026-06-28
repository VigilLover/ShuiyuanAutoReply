import os
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import MentionChatModel

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5"
MIMO_DEFAULT_THINKING = "enabled"
MIMO_DEFAULT_MAX_TOKENS: int | None = None
MIMO_DEFAULT_MAX_RETRIES = 3


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(sorted(allowed))}; got {value!r}."
        )
    return value


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


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


def _env_optional_positive_int(name: str, default: int | None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


def _mk_mimo_llm(api_key: str, model_name: str) -> ChatOpenAI:
    thinking = _env_choice(
        "MIMO_MENTION_THINKING",
        MIMO_DEFAULT_THINKING,
        {"enabled", "disabled"},
    )
    max_tokens = _env_optional_positive_int(
        "MIMO_MENTION_MAX_TOKENS",
        MIMO_DEFAULT_MAX_TOKENS,
    )
    max_retries = _env_positive_int(
        "MIMO_MENTION_MAX_RETRIES",
        MIMO_DEFAULT_MAX_RETRIES,
    )

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

    def __init__(self, model: ShuiyuanModel, username: str = "wolf_lumine"):
        super().__init__(model, username=username)

        api_key = os.getenv("MIMO_API_KEY")
        if not api_key:
            raise ValueError("Please set the MIMO_API_KEY environment variable.")

        model_name = os.getenv("MIMO_MENTION_MODEL", MIMO_DEFAULT_MODEL)
        self.llm = _mk_mimo_llm(api_key, model_name)
        self.supports_multimodal = True
        self.multimodal_search_image_limit = _env_positive_int(
            "MIMO_MULTIMODAL_MAX_SEARCH_IMAGES",
            2,
        )

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

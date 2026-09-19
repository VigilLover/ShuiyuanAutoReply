"""DeepSeek chat client construction and message-shape helpers.

Everything here is LangGraph-agnostic: building the ``ChatOpenAI`` client for
either API format, and converting stored image blocks into the shape each
format expects.  See https://api-docs.deepseek.com for the contracts.
"""

from __future__ import annotations

from typing import Any, Iterable

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.bootstrap.settings import DeepSeekApiFormat, ProviderSettings

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-flash"
# Connection-level retries only; the agent loop owns time budgets and never
# wants a silent second attempt after a slow request.
DEEPSEEK_DEFAULT_MAX_RETRIES = 1


def as_responses_image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a stored Chat/Vision image block to Responses ``input_image``."""
    block_type = block.get("type")
    if block_type == "input_image":
        return dict(block)
    if block_type == "file":
        return {"type": "input_image", "file_id": block["file_id"]}
    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            result = {"type": "input_image", "image_url": image_url.get("url", "")}
            if image_url.get("detail"):
                result["detail"] = image_url["detail"]
            return result
        return {"type": "input_image", "image_url": image_url}
    raise ValueError(f"Unsupported DeepSeek image block: {block_type!r}")


def _image_label(index: int, image: Any) -> str:
    label = image.description or image.source_url
    if image.source_kind in {"web_search", "forum_search"}:
        label = (
            f"{label}；展示标识 {image.artifact.uri}。"
            "最终回复需要展示此图时，只能把该展示标识作为图片地址"
        )
    return f"【图片 {index}：{label}】"


def build_deepseek_content(text: str, images: Iterable[Any]) -> list[dict[str, Any]]:
    """Chat Completions content: labeled images followed by the text."""
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        content.append({"type": "text", "text": _image_label(index, image)})
        content.append(image.content_block)
    if text:
        content.append({"type": "text", "text": text})
    return content


def build_deepseek_responses_content(
    text: str, images: Iterable[Any]
) -> list[dict[str, Any]]:
    """Responses API content: labeled ``input_image`` parts followed by the text."""
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        content.append({"type": "input_text", "text": _image_label(index, image)})
        content.append(as_responses_image_block(image.content_block))
    if text:
        content.append({"type": "input_text", "text": text})
    return content


def tool_output_blocks(content: Any) -> list[dict[str, Any]]:
    """Keep tool text while restricting output to Responses-compatible blocks."""
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "input_text", "text": item})
            elif isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                blocks.append({"type": "input_text", "text": str(item.get("text", ""))})
            elif isinstance(item, dict) and item.get("type") == "input_image":
                blocks.append(dict(item))
            else:
                blocks.append({"type": "input_text", "text": str(item)})
        return blocks
    return [{"type": "input_text", "text": str(content)}]


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that preserves DeepSeek thinking metadata.

    In thinking mode DeepSeek requires assistant ``reasoning_content`` to be sent
    back on every later Chat Completions request that carries tools. LangChain's
    generic adapter drops the field, so keep it in ``additional_kwargs`` and
    re-inject it into the payload.
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
            if isinstance(source_message, HumanMessage) and isinstance(
                source_message.content, list
            ):
                # DeepSeek's Vision endpoint accepts provider-specific ``file``
                # blocks that generic OpenAI adapters may otherwise normalize away.
                payload_message["content"] = source_message.content
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


def build_chat_model(
    api_key: str,
    model_name: str,
    settings: ProviderSettings | None = None,
    *,
    effort: str | None = None,
) -> ChatOpenAI:
    """Build the DeepSeek client for the configured API format.

    ``effort`` overrides the investigation effort from settings so the caller
    can hold one client per phase (tool planning vs. final answer).
    """
    current = settings or ProviderSettings()
    current.validate_deepseek_options()
    thinking = current.deepseek_thinking
    reasoning_effort = effort or current.deepseek_reasoning_effort
    max_tokens = current.deepseek_max_tokens

    common_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": (current.mention_base_url or "").strip().rstrip("/")
        or DEEPSEEK_BASE_URL,
        "max_retries": DEEPSEEK_DEFAULT_MAX_RETRIES,
        "timeout": float(current.deepseek_request_timeout),
    }
    if DeepSeekApiFormat(current.deepseek_api_format) is DeepSeekApiFormat.RESPONSES:
        kwargs = {
            **common_kwargs,
            "use_responses_api": True,
            "output_version": "v1",
            "reasoning": {
                "effort": reasoning_effort if thinking == "enabled" else "none"
            },
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    kwargs = {
        **common_kwargs,
        "extra_body": {"thinking": {"type": thinking}},
    }
    if thinking == "enabled":
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return DeepSeekChatOpenAI(**kwargs)

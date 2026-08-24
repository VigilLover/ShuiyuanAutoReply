from typing import Any, Iterable

import logging
from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.bootstrap.settings import DeepSeekApiFormat, ProviderSettings
from shuiyuan_auto_reply.application.ports.prompt import PromptScope

from .deepseek_vision import (
    MAX_IMAGES_PER_TURN,
    DeepSeekVisionMediaManager,
    build_deepseek_content,
)
from .mention_multimodal import extract_image_urls
from .mention_chat_model import MentionChatModel, MentionGraphState

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"
DEEPSEEK_DEFAULT_MAX_RETRIES = 3
DEEPSEEK_DEFAULT_THINKING = "enabled"
DEEPSEEK_DEFAULT_REASONING_EFFORT = "max"


def _as_responses_image_block(block: dict[str, Any]) -> dict[str, Any]:
    """Convert the stored Chat/Vision image block to Responses input format."""
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


def build_deepseek_responses_content(
    text: str, images: Iterable[Any]
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, 1):
        label = image.description or image.source_url
        if image.source_kind in {"web_search", "forum_search"}:
            label = (
                f"{label}；展示标识 {image.artifact.uri}。"
                "最终回复需要展示此图时，只能把该展示标识作为图片地址"
            )
        content.append({"type": "input_text", "text": f"【图片 {index}：{label}】"})
        content.append(_as_responses_image_block(image.content_block))
    if text:
        content.append({"type": "input_text", "text": text})
    return content


def _tool_output_blocks(content: Any) -> list[dict[str, Any]]:
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
            else:
                blocks.append({"type": "input_text", "text": str(item)})
        return blocks
    return [{"type": "input_text", "text": str(content)}]


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
            if isinstance(source_message, HumanMessage) and isinstance(
                source_message.content, list
            ):
                # DeepSeek's Vision endpoint accepts provider-specific ``file``
                # blocks that generic OpenAI adapters may otherwise normalize away.
                payload_message["content"] = source_message.content
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

    common_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": DEEPSEEK_BASE_URL,
        "max_retries": DEEPSEEK_DEFAULT_MAX_RETRIES,
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


class MentionDeepSeekModel(MentionChatModel):
    """Single-model DeepSeek V4 Flash Vision agent."""

    @property
    def uses_responses_api(self) -> bool:
        return (
            getattr(self, "api_format", DeepSeekApiFormat.CHAT_COMPLETIONS)
            is DeepSeekApiFormat.RESPONSES
        )

    def _get_multimodal_prompt_rules(self) -> str:
        return (
            "【原生视觉理解规则】\n"
            "1. 用户图片和工具结果图片会自动作为视觉输入附加，不要调用独立识图工具。\n"
            "2. 只有实际出现的图片可用于判断；工具结果没有图片时不要猜测画面。\n"
            "3. 图片标签只用于区分来源，回答时结合图片本身和相邻文字。\n\n"
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
        super().__init__(model, username=username, prompt_scope=prompt_scope, enabled_tools=enabled_tools, disabled_mcp_tools=disabled_mcp_tools, state_store=state_store, system_prompt_override=system_prompt_override)

        current = provider_settings or ProviderSettings()
        current.validate_deepseek_options()
        self.provider_settings = current
        self.api_format = DeepSeekApiFormat(current.deepseek_api_format)
        api_key = current.deepseek_api_key
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

        self.llm = _mk_deepseek_llm(
            api_key,
            DEEPSEEK_DEFAULT_MODEL,
            current,
        )
        if self.api_format is DeepSeekApiFormat.RESPONSES:
            self.hidden_provider_tools = [{"type": "web_search"}]
            self.provider_tool_choice = "auto"
        self.supports_multimodal = True
        self.multimodal_search_image_limit = MAX_IMAGES_PER_TURN
        self.uses_inspect_image_tool = False
        self.vision_media = DeepSeekVisionMediaManager(
            state_store=state_store,
            forum_model=model,
            api_key=api_key,
        )

    async def _load_current_images(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        images = list(state.get("image_inputs", []) or [])
        seen = {image.source_url for image in images}
        for attachment in state.get("request_attachments", ()) or ():
            if len(images) >= MAX_IMAGES_PER_TURN:
                break
            try:
                image = await self.vision_media.prepare_attachment(attachment)
            except Exception as exc:
                logging.warning("Failed to prepare user image: %s", exc)
                continue
            if image.source_url not in seen:
                seen.add(image.source_url)
                images.append(image)

        for url in extract_image_urls(state.get("conversation", "")):
            if len(images) >= MAX_IMAGES_PER_TURN or url in seen:
                break
            try:
                image = await self.vision_media.prepare_forum_url(
                    url,
                    conversation_id=state.get("conversation_id"),
                    source_kind="forum_post",
                    description="当前论坛帖子图片",
                )
            except Exception as exc:
                logging.warning("Failed to prepare current forum image %s: %s", url, exc)
                continue
            if image:
                seen.add(url)
                images.append(image)
        return {
            "supports_multimodal": True,
            "image_inputs": images,
            "input_visual_artifacts": [image.artifact for image in images],
        }

    async def _load_topic_context(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        result = await super()._load_topic_context(state)
        external_history = state.get("external_history") or ()
        historical_images = []
        for item in reversed(external_history):
            for attachment in reversed(getattr(item, "attachments", ()) or ()):
                if len(historical_images) >= MAX_IMAGES_PER_TURN:
                    break
                try:
                    historical_images.append(
                        await self.vision_media.prepare_attachment(attachment)
                    )
                except Exception as exc:
                    logging.warning("Failed to restore historical image: %s", exc)
            if len(historical_images) >= MAX_IMAGES_PER_TURN:
                break
        if historical_images:
            historical_images.reverse()
            content_builder = (
                build_deepseek_responses_content
                if self.uses_responses_api
                else build_deepseek_content
            )
            result["chat_history"].append(
                HumanMessage(
                    content=content_builder(
                        "以上是最近会话中的历史图片，仅在用户追问时结合使用。",
                        historical_images,
                    )
                )
            )
        return result

    async def _load_replied_post_images(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        images = list(state.get("image_inputs", []) or [])
        if not state.get("reply_to_post_number") or len(images) >= MAX_IMAGES_PER_TURN:
            return {"image_inputs": images}
        try:
            post = await self.model.get_post_details_by_post_number(
                state["topic_id"], state["reply_to_post_number"]
            )
        except Exception as exc:
            logging.warning("Failed to load replied-post images: %s", exc)
            return {"image_inputs": images}
        urls = list(getattr(post, "image_urls", []) or [])
        urls.extend(extract_image_urls(getattr(post, "raw", "")))
        urls.extend(extract_image_urls(getattr(post, "cooked", "")))
        seen = {image.source_url for image in images}
        for url in urls:
            if len(images) >= MAX_IMAGES_PER_TURN:
                break
            if url in seen:
                continue
            try:
                image = await self.vision_media.prepare_forum_url(
                    url,
                    conversation_id=state.get("conversation_id"),
                    source_kind="forum_post",
                    description="被回复论坛帖子图片",
                )
            except Exception as exc:
                logging.warning("Failed to prepare replied image %s: %s", url, exc)
                continue
            if image:
                seen.add(url)
                images.append(image)
        return {
            "image_inputs": images,
            "input_visual_artifacts": [image.artifact for image in images],
        }

    async def _prepare_messages(self, state: MentionGraphState) -> MentionGraphState:
        text = (
            "【用户当前发言】\n<user_post>\n"
            f"{state['conversation']}\n"
            "</user_post>"
        )
        images = state.get("image_inputs", []) or []
        if images and self.uses_responses_api:
            content = build_deepseek_responses_content(text, images)
        else:
            content = build_deepseek_content(text, images) if images else text
        return {"messages": [HumanMessage(content=content)]}

    async def _collect_tool_output_images(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        existing = list(state.get("image_inputs", []) or [])
        remaining = MAX_IMAGES_PER_TURN - len(existing)
        if remaining <= 0:
            return {"image_inputs": existing[:MAX_IMAGES_PER_TURN]}
        tool_messages: list[ToolMessage] = []
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "tool":
                break
            tool_messages.append(message)
        tool_messages.reverse()
        if not self.uses_responses_api:
            new_images = await self.vision_media.prepare_tool_output(
                tool_messages,
                conversation_id=state.get("conversation_id"),
                existing_urls={image.source_url for image in existing},
                limit=remaining,
            )
        else:
            new_images = []
            replacements: list[ToolMessage] = []
            existing_urls = {image.source_url for image in existing}
            for message in tool_messages:
                message_images = await self.vision_media.prepare_tool_output(
                    [message],
                    conversation_id=state.get("conversation_id"),
                    existing_urls=existing_urls,
                    limit=remaining - len(new_images),
                )
                if not message_images:
                    continue
                new_images.extend(message_images)
                existing_urls.update(image.source_url for image in message_images)
                output = _tool_output_blocks(message.content)
                for index, image in enumerate(message_images, 1):
                    output.append(
                        {
                            "type": "input_text",
                            "text": f"【工具图片 {index}：{image.description or image.source_url}】",
                        }
                    )
                    output.append(_as_responses_image_block(image.content_block))
                replacements.append(message.model_copy(update={"content": output}))
        if not new_images:
            return {"image_inputs": existing}
        visual_artifacts = list(state.get("response_visual_artifacts", []) or [])
        visual_artifacts.extend(image.artifact for image in new_images)
        result: MentionGraphState = {
            "image_inputs": existing + new_images,
            "response_visual_artifacts": visual_artifacts,
        }
        if self.uses_responses_api:
            result["messages"] = replacements
        else:
            result["messages"] = [
                HumanMessage(
                    content=build_deepseek_content(
                        "以上图片来自本轮工具返回，请结合对应来源文字继续回答。",
                        new_images,
                    )
                )
            ]
        return result

    async def _call_model(self, state: MentionGraphState) -> MentionGraphState:
        try:
            result = await super()._call_model(state)
        except Exception:
            url_images = [
                image
                for image in state.get("image_inputs", []) or []
                if image.content_block.get("type") == "image_url"
            ]
            if not url_images:
                raise
            logging.warning(
                "DeepSeek could not read %d image URL(s); retrying once with Files API",
                len(url_images),
            )
            replacements: dict[str, dict[str, Any]] = {}
            replaced_images = []
            for image in state.get("image_inputs", []) or []:
                block = image.content_block
                if block.get("type") == "image_url":
                    file_id = await self.vision_media.ensure_file_id(image.artifact)
                    stored_replacement = {"type": "file", "file_id": file_id}
                    replacement = (
                        {"type": "input_image", "file_id": file_id}
                        if self.uses_responses_api
                        else stored_replacement
                    )
                    url = str((block.get("image_url") or {}).get("url", ""))
                    replacements[url] = replacement
                    image = replace(image, content_block=stored_replacement)
                replaced_images.append(image)
            state["image_inputs"] = replaced_images
            for message in state.get("messages", []) or []:
                content = getattr(message, "content", None)
                if not isinstance(content, list):
                    continue
                rewritten = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {
                        "image_url",
                        "input_image",
                    }:
                        image_url = block.get("image_url")
                        url = str(
                            image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else image_url or ""
                        )
                        rewritten.append(replacements.get(url, block))
                    else:
                        rewritten.append(block)
                message.content = rewritten
            result = await super()._call_model(state)
        if self.uses_responses_api:
            response = (result.get("messages") or [None])[-1]
            await self._emit_native_web_search_events(response)
        return result

    async def _emit_native_web_search_events(self, response: Any) -> None:
        for item in getattr(response, "content", []) or []:
            item_type = (
                item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
            )
            if item_type != "web_search_call":
                continue
            await emit_event(
                "provider.tool.completed",
                {
                    "provider": "deepseek",
                    "name": "web_search",
                    "call_id": (
                        item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
                    ),
                    "status": (
                        item.get("status")
                        if isinstance(item, dict)
                        else getattr(item, "status", None)
                    ),
                },
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
                if (
                    isinstance(item, dict)
                    and item.get("type") in {"text", "output_text"}
                    and "text" in item
                ):
                    res += item["text"]
                if (
                    getattr(item, "type", None) in {"text", "output_text"}
                    and hasattr(item, "text")
                ):
                    res += item.text
                if isinstance(item, str):
                    res += item
        return res.strip()

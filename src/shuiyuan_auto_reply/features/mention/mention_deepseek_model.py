import logging
from dataclasses import replace
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.bootstrap.settings import DeepSeekApiFormat, ProviderSettings
from shuiyuan_auto_reply.infrastructure.llm.deepseek import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    DeepSeekChatOpenAI,
    as_responses_image_block,
    build_chat_model,
    build_deepseek_content,
    build_deepseek_responses_content,
    tool_output_blocks,
)
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .deepseek_vision import MAX_IMAGES_PER_TURN, DeepSeekVisionMediaManager
from .mention_chat_model import MentionChatModel, MentionGraphState
from .mention_multimodal import extract_image_urls

# Backwards-compatible aliases for tests and callers of the previous module layout.
_mk_deepseek_llm = build_chat_model
_as_responses_image_block = as_responses_image_block
_tool_output_blocks = tool_output_blocks

__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_DEFAULT_MODEL",
    "DeepSeekChatOpenAI",
    "MentionDeepSeekModel",
    "build_deepseek_content",
    "build_deepseek_responses_content",
]


class MentionDeepSeekModel(MentionChatModel):
    """Single-model DeepSeek V4 Flash Vision agent."""

    @property
    def uses_responses_api(self) -> bool:
        return (
            getattr(self, "api_format", DeepSeekApiFormat.CHAT_COMPLETIONS)
            is DeepSeekApiFormat.RESPONSES
        )

    def _get_multimodal_prompt_rules(self) -> str:
        # Non-empty enables the shared【图片理解】capability block; the text itself
        # is only used for legacy (non-managed) prompts.
        return (
            "【图片理解】\n"
            "1. 当前用户附带的图片自动进入视觉输入；工具结果中的图片只在显式请求时载入。\n"
            "2. 只有实际载入的图片可用于判断，图片标签只用于区分来源。\n\n"
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
        current.validate_deepseek_options()
        self.provider_settings = current
        self.api_format = DeepSeekApiFormat(current.deepseek_api_format)
        api_key = current.deepseek_api_key
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

        model_name = (current.deepseek_model or "").strip() or DEEPSEEK_DEFAULT_MODEL
        self.llm = build_chat_model(api_key, model_name, current)
        # The final answer never plans tool calls, so it can run at its own effort.
        self.llm_final = (
            self.llm
            if current.deepseek_final_reasoning_effort
            == current.deepseek_reasoning_effort
            else build_chat_model(
                api_key,
                model_name,
                current,
                effort=current.deepseek_final_reasoning_effort,
            )
        )
        self.supports_multimodal = True
        self.multimodal_search_image_limit = MAX_IMAGES_PER_TURN
        self.vision_media = DeepSeekVisionMediaManager(
            state_store=state_store,
            forum_model=model,
            api_key=api_key,
            base_url=(current.mention_base_url or "").strip().rstrip("/")
            or DEEPSEEK_BASE_URL,
        )

    async def _load_current_images(self, state: MentionGraphState) -> MentionGraphState:
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
                logging.warning(
                    "Failed to prepare current forum image %s: %s", url, exc
                )
                continue
            if image:
                seen.add(url)
                images.append(image)
        return {
            "supports_multimodal": True,
            "image_inputs": images,
            "input_visual_artifacts": [image.artifact for image in images],
        }

    async def _load_topic_context(self, state: MentionGraphState) -> MentionGraphState:
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
        existing = list(state.get("image_inputs", []) or [])
        return {"image_inputs": existing}

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
        remaining = min(4, MAX_IMAGES_PER_TURN - len(existing))
        if remaining <= 0:
            return {"image_inputs": existing[:MAX_IMAGES_PER_TURN]}
        tool_messages: list[ToolMessage] = []
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "tool":
                break
            tool_messages.append(message)
        tool_messages.reverse()
        tool_messages = [
            m
            for m in tool_messages
            if m.name in {"forum_read", "users", "web_read", "generate_image"}
        ]
        if not tool_messages:
            return {"image_inputs": existing}

        async def images_for(message: ToolMessage, limit: int, existing_urls: set):
            if message.name == "generate_image":
                artifact = getattr(message, "artifact", None)
                if artifact is None or limit <= 0:
                    return []
                preview = self.vision_media.prepare_generated(artifact)
                return [preview] if preview else []
            return await self.vision_media.prepare_tool_output(
                [message],
                conversation_id=state.get("conversation_id"),
                existing_urls=existing_urls,
                limit=limit,
            )

        new_images = []
        replacements: list[ToolMessage] = []
        existing_urls = {image.source_url for image in existing}
        for message in tool_messages:
            message_images = await images_for(
                message, remaining - len(new_images), existing_urls
            )
            if not message_images:
                continue
            new_images.extend(message_images)
            existing_urls.update(image.source_url for image in message_images)
            if self.uses_responses_api:
                output = tool_output_blocks(message.content)
                for index, image in enumerate(message_images, 1):
                    output.append(
                        {
                            "type": "input_text",
                            "text": f"【工具图片 {index}：{image.description or image.source_url}】",
                        }
                    )
                    output.append(as_responses_image_block(image.content_block))
                replacements.append(message.model_copy(update={"content": output}))
        if not new_images:
            return {"image_inputs": existing}
        visual_artifacts = list(state.get("response_visual_artifacts", []) or [])
        # Generated previews are already delivered through generated_artifacts.
        visual_artifacts.extend(
            image.artifact for image in new_images if image.source_kind != "generated"
        )
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
        except Exception as exc:
            url_images = [
                image
                for image in state.get("image_inputs", []) or []
                if image.content_block.get("type") == "image_url"
            ]
            # Only a provider rejection of the image payload justifies re-sending
            # through the Files API; timeouts and other failures must not double
            # the spend of an already slow round.
            if not url_images or getattr(exc, "status_code", None) != 400:
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
        return result

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
                if getattr(item, "type", None) in {"text", "output_text"} and hasattr(
                    item, "text"
                ):
                    res += item.text
                if isinstance(item, str):
                    res += item
        return res.strip()

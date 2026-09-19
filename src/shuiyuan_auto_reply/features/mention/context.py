"""Graph nodes that load context before the model runs: style, forum, memory, images."""

import logging

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage

from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.application.tool_results import current_turn
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .graph_state import MentionGraphState
from .shuiyuan_tools_objects import PostShort


class ContextMixin:
    async def _retrieve_style_context(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        try:
            persona = state.get("persona")
            if not persona:
                logging.warning(
                    "Mention graph has no persona in state; skipping style context retrieval"
                )
                return {"context": ""}

            style_items = await self.style_retriever.search(
                persona,
                state["conversation"],
                8,
            )
        except Exception as exc:
            logging.exception("Failed to retrieve style context; continuing without it")
            await emit_event(
                "context.style_failed",
                {"error": type(exc).__name__, "message": str(exc)[:500]},
            )
            return {"context": ""}

        context_text = "\n".join(item.text for item in style_items)
        await emit_event(
            "context.style_loaded",
            {"count": len(style_items), "persona": persona, "limit": 8},
        )
        logging.info(
            "Mention graph retrieved %d style document(s), persona=%s context_chars=%d",
            len(style_items),
            persona,
            len(context_text),
        )
        return {"context": context_text}

    async def _load_topic_context(self, state: MentionGraphState) -> MentionGraphState:
        external_history = state.get("external_history")
        if external_history is not None:
            history_obj = InMemoryChatMessageHistory()
            for item in external_history:
                # Stored replies carry the signature and auto-reply tag; strip
                # them so the model cannot copy the format into its own output.
                content = ShuiyuanModel.strip_forum_signature(item.content)
                if item.role == "user":
                    history_obj.add_user_message(content)
                elif item.role == "assistant":
                    history_obj.add_ai_message(content)
        else:
            history_obj = self.get_session_history(state["session_id"])
        topic_id = state.get("topic_id")
        if state.get("load_forum_context", True) and topic_id is not None:
            recent_msgs = await self.get_recent_msgs_context(
                topic_id, reply_to_post_number=state.get("reply_to_post_number")
            )
            await emit_event("context.forum_loaded", {"topic_id": topic_id})
        else:
            recent_msgs = "无近期回帖记录"
            await emit_event("context.forum_skipped", {})
        target_post = None
        if (
            state.get("load_forum_context", True)
            and topic_id is not None
            and state.get("reply_to_post_number")
        ):
            target_post = PostShort(
                await self.model.get_post_details_by_post_number(
                    topic_id, state["reply_to_post_number"]
                ),
                full=True,
            )
        turn = current_turn.get()
        if turn and target_post is not None:
            turn.observe(str(target_post), tool="forum_read")
        return {
            "target_post": target_post,
            "chat_history": history_obj.messages,
            "history_obj": history_obj,
            "recent_msgs": recent_msgs,
        }

    async def _load_long_term_memory(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        user = state["user"]
        memory_user_id = state.get("memory_user_id", user.id)
        memory_key = self.memory_model.memory_key(memory_user_id)
        memory_context = await self.memory_model.search_mention_memory(
            target_user_id=memory_user_id,
            query=state["conversation"],
            limit=self.memory_model.search_limit,
        )
        logging.info(
            "Mention graph loaded long-term memory: user_id=%s chars=%d preview=%r",
            memory_key,
            len(memory_context),
            memory_context[:256],
        )
        await emit_event("memory.loaded", {"chars": len(memory_context)})
        return {"long_term_memory": memory_context}

    async def _load_current_images(self, state: MentionGraphState) -> MentionGraphState:
        return {
            "supports_multimodal": bool(self.supports_multimodal),
            "image_inputs": list(state.get("image_inputs", []) or []),
        }

    async def _load_replied_post_images(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        return {"image_inputs": list(state.get("image_inputs", []) or [])}

    @staticmethod
    async def _prepare_messages(state: MentionGraphState) -> MentionGraphState:
        content = (
            "【用户当前发言】\n"
            "<user_post>\n"
            f"{state['conversation']}\n"
            "</user_post>"
        )
        return {"messages": [HumanMessage(content=content)]}

    async def get_recent_msgs_context(
        self,
        topic_id: int,
        limit: int = 8,
        *,
        reply_to_post_number: int | None = None,
        chain_depth: int = 3,
    ) -> str:
        """Recent posts in the topic plus the ancestors of the post being answered.

        A reply usually continues a specific thread inside a busy topic, so the
        last few posts alone often miss what is actually being discussed. The
        reply chain is followed upward through ``reply_to_post_number`` and the
        ancestors are prepended, oldest first, so the model sees the thread in
        reading order.
        """
        try:
            title, values, _next_offset, _has_more = (
                await self.model.read_topic_post_page(
                    topic_id,
                    offset=0,
                    limit=limit,
                    ascending=False,
                )
            )
            posts = [PostShort(post, title) for post in values]
        except Exception as exc:
            logging.warning("Failed to load recent forum context: %s", exc)
            return "无近期回帖记录"

        seen = {post.post_number for post in posts}
        chain: list[PostShort] = []
        number = reply_to_post_number
        try:
            for _ in range(chain_depth):
                if not number or number in seen:
                    break
                post = PostShort(
                    await self.model.get_post_details_by_post_number(topic_id, number),
                    title,
                )
                seen.add(post.post_number)
                chain.append(post)
                number = post.reply_to_post_number
        except Exception as exc:
            logging.warning("Failed to follow the reply chain: %s", exc)

        if not posts and not chain:
            return "无近期回帖记录"
        sections = []
        if chain:
            sections.append(
                "【被回复楼层的上文，从早到晚】\n"
                + "\n\n".join(str(post) for post in reversed(chain))
            )
        if posts:
            sections.append(
                "【话题最新回帖，从新到旧】\n"
                + "\n\n".join(str(post) for post in posts)
            )
        return "\n\n".join(sections)

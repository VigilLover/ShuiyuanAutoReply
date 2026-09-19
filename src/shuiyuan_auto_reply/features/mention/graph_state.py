"""The LangGraph state shape shared by every node in the mention chat graph."""

from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from shuiyuan_auto_reply.domain import ChatMessage, GeneratedImageArtifact
from shuiyuan_auto_reply.shuiyuan.objects import User


class MentionGraphState(TypedDict, total=False):
    persona: str
    target_post: object
    tool_validation_errors: dict[str, str]
    topic_id: Optional[int]
    session_id: int | str
    load_forum_context: bool
    memory_user_id: int | str
    reply_to_post_number: Optional[int]
    conversation: str
    user: User
    context: str
    long_term_memory: str
    chat_history: List[AnyMessage]
    recent_msgs: str
    raw_output: object
    final_text: str
    history_obj: InMemoryChatMessageHistory
    messages: Annotated[List[AnyMessage], add_messages]
    image_inputs: List[Any]
    supports_multimodal: bool
    external_history: tuple[ChatMessage, ...] | None
    generated_artifacts: list[GeneratedImageArtifact]
    request_attachments: tuple[object, ...]
    conversation_id: str | None
    input_visual_artifacts: list[object]
    response_visual_artifacts: list[object]

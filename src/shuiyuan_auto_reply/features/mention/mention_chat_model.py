import inspect
import json
import logging
import os
from abc import abstractmethod
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import ConfigDict
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from shuiyuan_auto_reply.embeddings import get_global_text_embeddings
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.domain import ChatMessage, GeneratedImageArtifact
from shuiyuan_auto_reply.infrastructure.retrieval import Neo4jStyleRetriever
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
from shuiyuan_auto_reply.openrouter.openrouter_model import (
    DEFAULT_OPENROUTER_MAX_RETRIES,
)
from shuiyuan_auto_reply.shuiyuan.objects import User
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from .image_generation import ImageGenerationService, create_image_generation_tool
from .chat_pipeline import ChatOrchestrator
from .mention_memory_model import MentionMemoryModel
from .mention_multimodal import (
    ImageInspectResult,
    MentionImageInput,
    build_mimo_content,
    collect_post_image_inputs,
    extract_image_urls,
    normalize_shuiyuan_image_url,
)
from .shuiyuan_tools_wrapper import ShuiyuanToolsWrapper


class MentionGraphState(TypedDict, total=False):
    persona: str
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
    history_obj: ChatMessageHistory
    messages: Annotated[List[AnyMessage], add_messages]
    image_inputs: List[MentionImageInput]
    supports_multimodal: bool
    external_history: tuple[ChatMessage, ...] | None
    generated_artifacts: list[GeneratedImageArtifact]


class FallbackLLM(BaseChatModel):
    """
    A BaseChatModel that tries a primary LLM first and falls back to a secondary
    LLM on failure.  Inherits from BaseChatModel so LangChain's create_tool_calling_agent
    accepts it, and bind_tools returns a RunnableLambda that preserves fallback logic
    through the pipe chain.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: object = None
    fallback: object = None

    def __init__(self, primary, fallback, **kwargs):
        super().__init__(primary=primary, fallback=fallback, **kwargs)

    def _generate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        try:
            return self.primary._generate(messages, stop, run_manager, **kwargs)
        except Exception:
            logging.warning("[FallbackLLM] Primary LLM failed, falling back to secondary...")
            return self.fallback._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: list,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        try:
            return await self.primary._agenerate(messages, stop, run_manager, **kwargs)
        except Exception:
            logging.warning("[FallbackLLM] Primary LLM failed, falling back to secondary...")
            return await self.fallback._agenerate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "fallback-llm"

    def bind_tools(self, tools, **kwargs):
        primary_bound = self.primary.bind_tools(tools, **kwargs)
        fallback_bound = self.fallback.bind_tools(tools, **kwargs)

        async def _afn(input, config=None, **kw):
            try:
                return await primary_bound.ainvoke(input, config, **kw)
            except Exception:
                logging.warning("[FallbackLLM] Primary LLM failed, falling back to secondary...")
                return await fallback_bound.ainvoke(input, config, **kw)

        def _fn(input, config=None, **kw):
            try:
                return primary_bound.invoke(input, config, **kw)
            except Exception:
                logging.warning("[FallbackLLM] Primary LLM failed, falling back to secondary...")
                return fallback_bound.invoke(input, config, **kw)

        return RunnableLambda(_fn, afunc=_afn)


class MentionChatModel:
    """
    A model for generating responses in a forum context,
    specifically designed to mimic the style of a specific user persona.
    It integrates with a vector database for retrieving relevant historical messages and recent posts,
    and can utilize tools provided by an MCP server as well as custom tools defined in the ShuiyuanModel.
    """

    def __init__(
        self,
        model: ShuiyuanModel,
        username="wolf_lumine",
        *,
        prompt_scope: PromptScope = PromptScope.FORUM,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
    ):
        # The llm model should be defined in the subclass
        self.llm: BaseChatModel
        self.username = username
        self.provider_settings = ProviderSettings()
        # The embedding model used in this application
        self.embeddings = get_global_text_embeddings()

        prompt_repository = FilePromptRepository()
        capabilities = (
            {"multimodal"} if self._get_multimodal_prompt_rules() else set()
        )
        system_prompt = prompt_repository.load(
            username, capabilities, prompt_scope
        ).system_prompt
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
        self.prompt_scope = prompt_scope
        self.enabled_tools = enabled_tools
        self.disabled_mcp_tools = disabled_mcp_tools or set()
        self.state_store = state_store

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        # Initialize message histories
        self._histories: Dict[int | str, ChatMessageHistory] = {}

        # LangGraph runtime objects are initialized after subclass sets self.llm.
        self.graph: Optional[CompiledStateGraph] = None
        self.llm_with_tools = None
        self.openai_tools: List[Dict[str, Any]] = []
        self.tools: List[BaseTool] = []
        self.memory_model = MentionMemoryModel(self.embeddings)
        self.model = model
        self.supports_multimodal = False
        self.multimodal_search_image_limit = 0
        self.style_retriever = Neo4jStyleRetriever()
        self.pipeline = ChatOrchestrator(self)

    def _get_multimodal_prompt_rules(self) -> str:
        """图片理解相关的系统提示规则。子类覆盖以添加多模态图片理解规则。"""
        return ""

    def get_session_history(self, session_id: int | str) -> ChatMessageHistory:
        history = self._histories.setdefault(session_id, ChatMessageHistory())
        self._trim_session_history(history)
        return history

    @staticmethod
    def _preview_text(value: object, limit: Optional[int] = 512) -> str:
        return str(value).replace("\n", "\\n")[:limit]

    @classmethod
    def _prompt_event_value(cls, value: object, *, depth: int = 0) -> object:
        """Make model input inspectable without persisting embedded image bytes.

        This deliberately serializes only values that are actually part of the
        model input.  Provider reasoning fields are not inspected or emitted.
        """
        if depth > 8:
            return "[内容层级过深，已省略]"
        if isinstance(value, str):
            if value.lstrip().lower().startswith("data:"):
                return "[内嵌图片数据已省略]"
            return value
        if isinstance(value, dict):
            serialized: dict[str, object] = {}
            for key, item in value.items():
                normalized = str(key).lower().replace("-", "_")
                if normalized in {
                    "authorization",
                    "cookie",
                    "set_cookie",
                    "api_key",
                    "apikey",
                    "secret",
                } or normalized.endswith("_api_key"):
                    serialized[str(key)] = "[REDACTED]"
                else:
                    serialized[str(key)] = cls._prompt_event_value(
                        item, depth=depth + 1
                    )
            return serialized
        if isinstance(value, (list, tuple)):
            return [
                cls._prompt_event_value(item, depth=depth + 1) for item in value
            ]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return str(value)

    @classmethod
    def _prompt_messages_for_event(cls, prompt_value: object) -> list[dict[str, object]]:
        """Serialize the role/content payload sent to the chat model for the UI."""
        to_messages = getattr(prompt_value, "to_messages", None)
        messages = to_messages() if callable(to_messages) else []
        serialized: list[dict[str, object]] = []
        for message in messages:
            item: dict[str, object] = {
                "role": getattr(message, "type", type(message).__name__),
                "content": cls._prompt_event_value(
                    getattr(message, "content", "")
                ),
            }
            name = getattr(message, "name", None)
            if name:
                item["name"] = str(name)
            tool_call_id = getattr(message, "tool_call_id", None)
            if tool_call_id:
                item["tool_call_id"] = str(tool_call_id)
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                item["tool_calls"] = cls._prompt_event_value(tool_calls)
            serialized.append(item)
        return serialized

    @staticmethod
    def _trim_session_history(history: ChatMessageHistory) -> None:
        max_history_turns = 8
        turns: List[List[AnyMessage]] = []
        current_turn: List[AnyMessage] = []

        for message in history.messages:
            if getattr(message, "type", None) == "human":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [message]
            elif current_turn:
                current_turn.append(message)
            else:
                turns.append([message])

        if current_turn:
            turns.append(current_turn)

        if len(turns) > max_history_turns:
            history.messages = [
                message for turn in turns[-max_history_turns:] for message in turn
            ]

    @staticmethod
    def _extract_tool_call_name_args(tool_call: object) -> Tuple[str, object]:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function")
            if isinstance(function_payload, dict):
                tool_name = function_payload.get("name") or tool_call.get("name")
                tool_args = function_payload.get("arguments", {})
            else:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", tool_call.get("arguments", {}))
            return tool_name or "<unknown>", tool_args

        return getattr(tool_call, "name", "<unknown>"), getattr(tool_call, "args", {})

    @staticmethod
    def _serialize_tool_args(tool_args: object) -> str:
        if isinstance(tool_args, str):
            text = tool_args
        else:
            try:
                text = json.dumps(tool_args, ensure_ascii=False, default=str)
            except TypeError:
                text = str(tool_args)

        return text.replace("\n", "\\n")

    @staticmethod
    def _env_positive_int(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            logging.warning("Invalid integer for %s=%r, using %s", name, raw_value, default)
            return default
        return value if value > 0 else default

    @staticmethod
    def _existing_image_source_urls(state: MentionGraphState) -> list[str]:
        return [
            image.source_url
            for image in state.get("image_inputs", []) or []
            if getattr(image, "source_url", None)
        ]

    @staticmethod
    def _existing_image_byte_count(state: MentionGraphState) -> int:
        return sum(
            max(0, getattr(image, "byte_count", 0) or 0)
            for image in state.get("image_inputs", []) or []
        )

    @staticmethod
    def _artifact_posts(artifact: object) -> list[object]:
        if artifact is None:
            return []
        if isinstance(artifact, dict):
            if artifact.get("source") != "inspect_image":
                return []
            return [artifact]
        if isinstance(artifact, (list, tuple, set)):
            return [
                item
                for item in artifact
                if getattr(item, "source", None) == "inspect_image"
                or (isinstance(item, dict) and item.get("source") == "inspect_image")
            ]
        if getattr(artifact, "source", None) != "inspect_image":
            return []
        return [artifact]

    def clear_session_history(self, session_id: int | str) -> None:
        self._histories.pop(session_id, None)

    @staticmethod
    async def _load_mcp_tools(url: str) -> List[StructuredTool]:
        """
        Load tools from MCP Server and convert them to LangChain StructuredTool.
        """
        logging.info("Loading MCP tools from %s", url)

        # Get the list of tools from MCP Server
        client = MultiServerMCPClient(
            {
                "default": {
                    "transport": "sse",
                    "url": url,
                    "sse_read_timeout": 900,  # Image generation may take long
                }
            }
        )
        mcp_tools = await client.get_tools()

        # Log all tools loaded
        logging.info(
            "Loaded %d MCP tool(s): %s",
            len(mcp_tools),
            ", ".join(tool.name for tool in mcp_tools),
        )
        return mcp_tools

    def _load_shuiyuan_tools(self) -> List[StructuredTool]:
        # 函数名 → 工具名映射：工具名用短名，避免 LLM 记不住长名而调用错误
        _TOOL_NAMES = {
            "search_user_by_term": "search_user",
            "search_user_by_user_id": "search_user_by_id",
            "search_post_details_by_optional_username_topic": "search_posts",
            "query_recent_posts_by_topic_id": "recent_posts",
            "search_post_details_by_time_range_and_topic": "search_posts_by_time",
            "get_post_details_by_post_number": "get_post",
        }

        tools_wrapper = ShuiyuanToolsWrapper(self.model)
        tools = []
        for func_name, tool_name in _TOOL_NAMES.items():
            func = getattr(tools_wrapper, func_name)
            if callable(func):
                tools.append(
                    StructuredTool.from_function(
                        coroutine=func,
                        name=tool_name,
                        description=inspect.getdoc(func)
                        or f"Tool for calling {func_name}",
                    )
                )

        if getattr(self, "supports_multimodal", False):
            async def inspect_image(image_url: str, description: str = "") -> tuple[str, ImageInspectResult]:
                """
                Read a Shuiyuan image or user avatar URL for MiMo multimodal understanding.

                Use this when you need to understand the visual content of an image
                from a post search result, a quoted Shuiyuan image URL, or a user avatar.
                The URL must be a Shuiyuan upload short URL, upload:// URL, or Shuiyuan
                user_avatar URL. Do not use this for external website images.

                :param image_url: Shuiyuan image or avatar URL to inspect.
                :param description: Optional description of this image (e.g. which user's
                    avatar this is). Use this when inspecting multiple images to help the
                    model distinguish them.
                """
                normalized = normalize_shuiyuan_image_url(image_url)
                if normalized is None:
                    return (
                        "图片读取失败：inspect_image 只支持水源 upload://、short-url 或 user_avatar 图片 URL。",
                        ImageInspectResult(image_urls=[], description=description),
                    )
                return (
                    "图片已读取，将在下一轮结合该图片回答。",
                    ImageInspectResult(image_urls=[normalized], description=description),
                )

            tools.append(
                StructuredTool.from_function(
                    coroutine=inspect_image,
                    name="inspect_image",
                    description=inspect.getdoc(inspect_image)
                    or "读取水源图片供 MiMo 多模态理解。",
                    response_format="content_and_artifact",
                )
            )

        # 注册图片生成工具 (本地实现, 生成后自动上传水源并返回 Markdown)
        if getattr(self, "state_store", None) is not None:
            gen_img_func = ImageGenerationService(
                self.model, self.state_store
            ).generate
        else:
            # Compatibility path for direct legacy construction and its snapshots.
            gen_img_func = create_image_generation_tool(self.model)
        tools.append(
            StructuredTool.from_function(
                coroutine=gen_img_func,
                name="generate_image",
                description=inspect.getdoc(gen_img_func)
                or "根据文字描述生成图片并保存为本地 Artifact.",
                response_format=(
                    "content_and_artifact" if getattr(self, "state_store", None) is not None else "content"
                ),
            )
        )

        logging.info(
            "Loaded %d Shuiyuan tool(s): %s",
            len(tools),
            ", ".join(tool.name for tool in tools),
        )
        return tools

    async def initialize_agent(self):
        logging.info("Initializing mention LangGraph agent")

        # MCP tools added here
        mcp_tools = []
        mcp_server_url = self.provider_settings.mcp_server_url

        if mcp_server_url:
            logging.info(f"==> [MCP] Attempting connection to MCP server at URL: {mcp_server_url}")
            # Create MCP streams and session, then load tools from it
            try:
                mcp_tools = await self._load_mcp_tools(mcp_server_url)
            except Exception as e:
                logging.error(
                    f"==> [MCP] Failed to connect to MCP Server at {mcp_server_url}: {e}"
                )
        else:
            logging.info("MCP_SERVER_URL is not set; skipping MCP tools")

        # Shuiyuan-specific tools added here
        shuiyuan_tools = self._load_shuiyuan_tools()
        logging.info(f"==> [Shuiyuan Tools Loaded]: {[t.name for t in shuiyuan_tools]}")

        # 互联网搜索工具（替代 native web_search，DeepSeek/Tongyi 不支持原生搜索）
        ddg_search_tool = DuckDuckGoSearchResults(
            name="internet_search",
            description="Use this tool to search the internet for up-to-date information.",
        )

        # LangMem persistent memory tools added here if configured.
        await self.memory_model.initialize()
        memory_tools = self.memory_model.tools

        # MCP uses an independent deny-list so newly discovered MCP tools are
        # enabled by default. Other tools retain the existing allow-list behavior.
        enabled_mcp_tools = [
            tool for tool in mcp_tools if tool.name not in self.disabled_mcp_tools
        ]
        other_function_like_tools = shuiyuan_tools + memory_tools
        tool_catalog = [
            {"name": tool.name, "source": "mcp"} for tool in mcp_tools
        ] + [
            {"name": tool.name, "source": "forum-read/image"}
            for tool in shuiyuan_tools
        ] + [
            {"name": tool.name, "source": "memory"} for tool in memory_tools
        ] + [
            {"name": str(tool.get("type")), "source": "provider-native"}
            for tool in self.openai_tools
            if tool.get("type")
        ]
        for item in tool_catalog:
            if item["source"] == "mcp":
                item["enabled"] = item["name"] not in self.disabled_mcp_tools
            else:
                item["enabled"] = (
                    self.enabled_tools is None or item["name"] in self.enabled_tools
                )
        if self.state_store is not None:
            await self.state_store.replace_tool_catalog(
                self.prompt_scope.value, tool_catalog
            )
        if self.enabled_tools is not None:
            other_function_like_tools = [
                tool for tool in other_function_like_tools
                if tool.name in self.enabled_tools
            ]
            self.openai_tools = [
                tool for tool in self.openai_tools
                if str(tool.get("type", "")) in self.enabled_tools
            ]
        all_function_like_tools = enabled_mcp_tools + other_function_like_tools
        all_tools = all_function_like_tools + self.openai_tools
        self.tools = all_function_like_tools
        logging.info(
            "Binding LLM with %d function-like tool(s), %d memory tool(s), "
            "and %d native OpenAI tool(s)",
            len(all_function_like_tools),
            len(memory_tools),
            len(self.openai_tools),
        )
        self.llm_with_tools = self.llm.bind_tools(all_tools).with_retry(
            stop_after_attempt=DEFAULT_OPENROUTER_MAX_RETRIES
        )
        self.graph = self._build_graph()
        logging.info("Mention LangGraph agent initialized")

    def _build_graph(self) -> CompiledStateGraph:
        logging.info("Building mention LangGraph workflow")

        # Create the tool node with all tools
        tool_node = ToolNode(self.tools, handle_tool_errors=True).with_retry(
            stop_after_attempt=DEFAULT_OPENROUTER_MAX_RETRIES
        )

        # Create the state graph and define the workflow
        workflow = StateGraph(MentionGraphState)
        workflow.add_node("retrieve_style_context", self.pipeline.style)
        workflow.add_node("load_topic_context", self.pipeline.channel)
        workflow.add_node("load_long_term_memory", self.pipeline.memory)
        if self.supports_multimodal:
            workflow.add_node("load_current_images", self.pipeline.multimodal.current)
            workflow.add_node("load_replied_post_images", self.pipeline.multimodal.replied)
        workflow.add_node("prepare_messages", self.pipeline.messages)
        workflow.add_node("call_model", self.pipeline.runtime)
        workflow.add_node("log_tool_calls", self._log_tool_calls)
        workflow.add_node("validate_tool_calls", self.pipeline.tool_validator)
        workflow.add_node("tools", tool_node)
        workflow.add_node("log_tool_outputs", self._log_tool_outputs)
        if self.supports_multimodal:
            workflow.add_node(
                "collect_tool_output_images", self.pipeline.multimodal.tool_outputs
            )
        workflow.add_node("finalize_response", self.pipeline.response)
        workflow.add_node("save_history", self.pipeline.history)

        # Define the workflow edges and conditions
        workflow.set_entry_point("retrieve_style_context")
        workflow.add_edge("retrieve_style_context", "load_topic_context")
        workflow.add_edge("load_topic_context", "load_long_term_memory")
        if self.supports_multimodal:
            workflow.add_edge("load_long_term_memory", "load_current_images")
            workflow.add_edge("load_current_images", "load_replied_post_images")
            workflow.add_edge("load_replied_post_images", "prepare_messages")
        else:
            workflow.add_edge("load_long_term_memory", "prepare_messages")
        workflow.add_edge("prepare_messages", "call_model")
        workflow.add_conditional_edges(
            "call_model",
            tools_condition,
            {"tools": "log_tool_calls", END: "finalize_response"},
        )
        workflow.add_edge("log_tool_calls", "validate_tool_calls")
        workflow.add_conditional_edges(
            "validate_tool_calls",
            self._has_valid_tool_calls,
            {"tools": "tools", "call_model": "call_model"},
        )
        workflow.add_edge("tools", "log_tool_outputs")
        if self.supports_multimodal:
            workflow.add_edge("log_tool_outputs", "collect_tool_output_images")
            workflow.add_edge("collect_tool_output_images", "call_model")
        else:
            workflow.add_edge("log_tool_outputs", "call_model")
        workflow.add_edge("finalize_response", "save_history")
        workflow.add_edge("save_history", END)

        # Whether to enable memory system
        if self.memory_model.enabled:
            compiled_graph = workflow.compile(store=self.memory_model.store)
        else:
            compiled_graph = workflow.compile()

        logging.info("Mention LangGraph workflow built")
        return compiled_graph

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
            history_obj = ChatMessageHistory()
            for item in external_history:
                if item.role == "user":
                    history_obj.add_user_message(item.content)
                elif item.role == "assistant":
                    history_obj.add_ai_message(item.content)
        else:
            history_obj = self.get_session_history(state["session_id"])
        topic_id = state.get("topic_id")
        if state.get("load_forum_context", True) and topic_id is not None:
            recent_msgs = await self.get_recent_msgs_context(topic_id)
            await emit_event("context.forum_loaded", {"topic_id": topic_id})
        else:
            recent_msgs = "无近期回帖记录"
            await emit_event("context.forum_skipped", {})
        return {
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
        supports_multimodal = bool(self.supports_multimodal)
        existing_images = list(state.get("image_inputs", []) or [])
        if not supports_multimodal:
            return {
                "supports_multimodal": False,
                "image_inputs": existing_images,
            }

        max_images = self._env_positive_int("MIMO_MULTIMODAL_MAX_IMAGES", 4)
        if len(existing_images) >= max_images:
            return {
                "supports_multimodal": True,
                "image_inputs": existing_images[:max_images],
            }

        post = SimpleNamespace(
            raw=state.get("conversation", ""),
            cooked="",
            image_urls=extract_image_urls(state.get("conversation", "")),
        )
        new_images = await collect_post_image_inputs(
            [post],
            shuiyuan_model=self.model,
            origin="current_post",
            max_images=max_images - len(existing_images),
            existing_urls=self._existing_image_source_urls(state),
            existing_byte_count=self._existing_image_byte_count(state),
        )
        return {
            "supports_multimodal": True,
            "image_inputs": existing_images + new_images,
        }

    async def _load_replied_post_images(self, state: MentionGraphState) -> MentionGraphState:
        existing_images = list(state.get("image_inputs", []) or [])
        if not state.get("supports_multimodal") or not state.get("reply_to_post_number"):
            return {"image_inputs": existing_images}

        max_images = self._env_positive_int("MIMO_MULTIMODAL_MAX_IMAGES", 4)
        if len(existing_images) >= max_images:
            return {"image_inputs": existing_images[:max_images]}

        try:
            replied_post = await self.model.get_post_details_by_post_number(
                state["topic_id"],
                state["reply_to_post_number"],
            )
        except Exception:
            logging.exception(
                "Failed to load replied post images: topic_id=%s post_number=%s",
                state.get("topic_id"),
                state.get("reply_to_post_number"),
            )
            return {"image_inputs": existing_images}

        new_images = await collect_post_image_inputs(
            [replied_post],
            shuiyuan_model=self.model,
            origin="replied_post",
            max_images=max_images - len(existing_images),
            existing_urls=self._existing_image_source_urls(state),
            existing_byte_count=self._existing_image_byte_count(state),
        )
        return {"image_inputs": existing_images + new_images}

    @staticmethod
    async def _prepare_messages(state: MentionGraphState) -> MentionGraphState:
        content = (
            "【用户当前发言】\n"
            "<user_post>\n"
            f"{state['conversation']}\n"
            "</user_post>"
        )
        if state.get("supports_multimodal") and state.get("image_inputs"):
            return {
                "messages": [
                    HumanMessage(
                        content=build_mimo_content(content, state.get("image_inputs", []))
                    )
                ]
            }
        return {"messages": [HumanMessage(content=content)]}

    @staticmethod
    async def _log_tool_calls(state: MentionGraphState) -> MentionGraphState:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []

        for tool_call in tool_calls:
            tool_name, tool_args = MentionChatModel._extract_tool_call_name_args(
                tool_call
            )

            logging.info(
                "Mention graph tool call: name=%s args=%s",
                tool_name,
                MentionChatModel._serialize_tool_args(tool_args),
            )
            await emit_event(
                "tool.started",
                {
                    "name": tool_name,
                    "arguments": MentionChatModel._prompt_event_value(tool_args),
                },
            )

        return {}

    async def _validate_tool_calls(self, state: MentionGraphState) -> MentionGraphState:
        """校验工具调用: 过滤掉幻觉的工具名和缺少必填参数的工具调用。

        对于无效调用，生成合成 ToolMessage 错误作为反馈，
        让 LLM 在下一轮知道调用失败的原因并自行纠正。
        只有合法调用才会传递到 ToolNode 真正执行。
        """
        last_message = state["messages"][-1]
        tool_calls = list(getattr(last_message, "tool_calls", []) or [])

        if not tool_calls:
            return {}

        valid_tool_names = {tool.name for tool in self.tools}
        valid_calls: list[dict] = []
        error_messages: list[ToolMessage] = []

        for tc in tool_calls:
            if isinstance(tc, dict):
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                call_id = tc.get("id", "")
            else:
                tool_name = getattr(tc, "name", "")
                tool_args = getattr(tc, "args", {})
                call_id = getattr(tc, "id", "")

            # 检查 1: 工具名是否存在
            if tool_name not in valid_tool_names:
                logging.warning(
                    "Filtering out hallucinated tool call: name=%s id=%s",
                    tool_name, call_id,
                )
                error_messages.append(ToolMessage(
                    content=(
                        f"错误: 工具 '{tool_name}' 不存在。"
                        f"可用的工具有: {', '.join(sorted(valid_tool_names))}。"
                        f"请使用正确的工具名称重试。"
                    ),
                    tool_call_id=call_id or f"invalid_{len(error_messages)}",
                ))
                continue

            # 检查 2: generate_image 必须有合法 prompt
            if tool_name == "generate_image":
                prompt = str((tool_args or {}).get("prompt", "")).strip()
                reject_reason: str | None = None
                if not prompt:
                    reject_reason = "generate_image 工具需要提供 'prompt' 参数。请用纯中文详细描述要生成的图片内容。"
                elif len(prompt) < 10:
                    reject_reason = (
                        f"generate_image 的 prompt 过短（仅 {len(prompt)} 个字符），"
                        "请提供至少 10 个字符的详细图片描述。"
                    )
                elif prompt.isdigit():
                    reject_reason = (
                        "generate_image 的 prompt 不能为纯数字。"
                        "请用纯中文详细描述要生成的图片内容。"
                    )
                elif len(set(prompt)) <= 2:
                    reject_reason = (
                        "generate_image 的 prompt 无意义（字符种类过少）。"
                        "请用纯中文详细描述要生成的图片内容。"
                    )
                if reject_reason is not None:
                    logging.warning(
                        "Filtering out generate_image call with invalid prompt=%r, id=%s",
                        prompt[:80],
                        call_id,
                    )
                    error_messages.append(ToolMessage(
                        content=f"错误: {reject_reason}",
                        tool_call_id=call_id or f"invalid_prompt_{len(error_messages)}",
                    ))
                    continue

            valid_calls.append(tc)

        # 如果有无效调用，替换最后一条 AIMessage 为仅含有效 tool_calls 的版本
        if len(valid_calls) != len(tool_calls):
            new_aimessage = AIMessage(
                content=getattr(last_message, "content", "") or "",
                tool_calls=valid_calls,
                id=getattr(last_message, "id", ""),
                name=getattr(last_message, "name", None),
                additional_kwargs=dict(
                    getattr(last_message, "additional_kwargs", {}) or {}
                ),
                response_metadata=dict(
                    getattr(last_message, "response_metadata", {}) or {}
                ),
                usage_metadata=getattr(last_message, "usage_metadata", None),
            )
            return {
                "messages": [
                    RemoveMessage(id=getattr(last_message, "id", "")),
                    new_aimessage,
                ]
                + error_messages
            }

        return {}

    def _has_valid_tool_calls(self, state: MentionGraphState) -> str:
        """条件路由: 验证后是否还有合法工具调用需要执行。

        返回 "tools" → ToolNode 执行合法调用
        返回 "call_model" → 所有调用都被过滤了, 让 LLM 看到错误并纠正
        """
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []
        return "tools" if tool_calls else "call_model"

    @staticmethod
    async def _log_tool_outputs(state: MentionGraphState) -> MentionGraphState:
        tool_messages = []
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "tool":
                break
            tool_messages.append(message)

        tool_messages.reverse()
        generated = list(state.get("generated_artifacts", []) or [])
        for message in tool_messages:
            logging.info(
                "Mention graph tool output: name=%s content=%s",
                getattr(message, "name", "<unknown>"),
                MentionChatModel._preview_text(getattr(message, "content", message)),
            )
            event_type = (
                "tool.failed"
                if getattr(message, "status", None) == "error"
                else "tool.completed"
            )
            await emit_event(
                event_type,
                {"name": getattr(message, "name", "<unknown>"), "output": MentionChatModel._preview_text(getattr(message, "content", message), 2000)},
            )
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, GeneratedImageArtifact):
                generated.append(artifact)
                await emit_event(
                    "image.generated",
                    {"artifact_id": artifact.artifact_id, "byte_count": artifact.byte_count},
                )

        return {"generated_artifacts": generated}

    async def _collect_tool_output_images(self, state: MentionGraphState) -> MentionGraphState:
        existing_images = list(state.get("image_inputs", []) or [])
        if not state.get("supports_multimodal") or self.multimodal_search_image_limit <= 0:
            return {"image_inputs": existing_images}

        tool_messages = []
        for message in reversed(state.get("messages", [])):
            if getattr(message, "type", None) != "tool":
                break
            tool_messages.append(message)
        tool_messages.reverse()

        tool_posts: list[object] = []
        for message in tool_messages:
            tool_posts.extend(self._artifact_posts(getattr(message, "artifact", None)))

        if not tool_posts:
            return {"image_inputs": existing_images}

        max_total_images = self._env_positive_int("MIMO_MULTIMODAL_MAX_IMAGES", 4)
        remaining_total = max_total_images - len(existing_images)
        if remaining_total <= 0:
            return {"image_inputs": existing_images[:max_total_images]}

        max_search_images = min(self.multimodal_search_image_limit, remaining_total)
        new_images = await collect_post_image_inputs(
            tool_posts,
            shuiyuan_model=self.model,
            origin="tool_output",
            max_images=max_search_images,
            existing_urls=self._existing_image_source_urls(state),
            existing_byte_count=self._existing_image_byte_count(state),
        )
        if not new_images:
            return {"image_inputs": existing_images}

        image_message = HumanMessage(
            content=build_mimo_content(
                "以上图片来自 inspect_image 工具调用。如有描述标注，请根据标注区分不同图片的归属（如用户头像对应的用户）。",
                new_images,
            )
        )
        return {
            "image_inputs": existing_images + new_images,
            "messages": [image_message],
        }

    @staticmethod
    def _build_tool_call_history_summary(messages: List[AnyMessage]) -> Optional[str]:
        tool_history_prefix = "【历史工具调用记录】"
        entries = []
        for message in messages:
            tool_calls = getattr(message, "tool_calls", []) or []
            for tool_call in tool_calls:
                tool_name, tool_args = MentionChatModel._extract_tool_call_name_args(
                    tool_call
                )
                entries.append(
                    f"{len(entries) + 1}. {tool_name} 参数: "
                    f"{MentionChatModel._serialize_tool_args(tool_args)}"
                )

        if not entries:
            return None

        return (
            f"{tool_history_prefix}\n"
            "以下是上一轮实际发生过的工具调用参数摘要，只用于连续对话参考，不要向用户复述。\n"
            + "\n".join(entries)
            + "\n工具返回值未写入历史；历史里的图片链接只代表过去结果。"
            "如本轮需要生成或修改图片，必须重新调用图片生成工具，不能编造图片URL。"
        )

    _MAX_TOOL_LOOP_MESSAGES = 20

    @staticmethod
    def _trim_tool_loop_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
        """裁剪工具调用循环中累积的消息，防止上下文膨胀导致模型退化。

        当消息数超过 _MAX_TOOL_LOOP_MESSAGES 时，保留前几条和最近的消息，
        中间的旧工具结果用摘要替换，避免在后续 LLM 调用中传递完整历史。
        """
        if len(messages) <= MentionChatModel._MAX_TOOL_LOOP_MESSAGES:
            return list(messages)

        keep_head = 3
        keep_tail = MentionChatModel._MAX_TOOL_LOOP_MESSAGES - keep_head - 1
        trimmed = list(messages[:keep_head])
        trimmed.append(
            HumanMessage(
                content=(
                    f"[系统提示: 已省略中间 {len(messages) - keep_head - keep_tail} 条历史消息。"
                    "请根据最近的上下文继续完成任务。]"
                )
            )
        )
        trimmed.extend(messages[-keep_tail:])
        logging.info(
            "Trimmed tool-loop messages: %d → %d messages",
            len(messages),
            len(trimmed),
        )
        return trimmed

    async def _call_model(self, state: MentionGraphState) -> MentionGraphState:
        if self.llm_with_tools is None:
            raise RuntimeError("MentionChatModel LLM is not initialized.")

        user = state["user"]
        prompt_value = self.prompt.invoke(
            {
                "topic_id": state["topic_id"],
                "reply_to_post_number": state["reply_to_post_number"],
                "user_id": user.id,
                "username": user.username,
                "name": user.name or "",
                "context": state.get("context", ""),
                "long_term_memory": state.get("long_term_memory", "无相关长期记忆"),
                "chat_history": state.get("chat_history", []),
                "recent_msgs": state.get("recent_msgs", "无近期回帖记录"),
                "messages": self._trim_tool_loop_messages(state.get("messages", [])),
            }
        )
        prompt_messages = self._prompt_messages_for_event(prompt_value)
        await emit_event(
            "model.prompt_prepared",
            {
                "scope": self.prompt_scope.value,
                "message_count": len(prompt_messages),
                "messages": prompt_messages,
            },
        )
        await emit_event("model.started", {})
        response = await self.llm_with_tools.ainvoke(prompt_value)
        usage = getattr(response, "usage_metadata", None) or {}
        await emit_event("model.completed", {"usage": usage})
        if usage:
            await emit_event("usage.recorded", usage)
        if not getattr(response, "content", None) and not getattr(response, "tool_calls", None):
            logging.warning(
                "Model returned empty AIMessage (no content, no tool_calls). "
                "message_keys=%s",
                [k for k in response.__dict__ if not k.startswith("_")],
            )
        return {"messages": [response]}

    async def _finalize_response(self, state: MentionGraphState) -> MentionGraphState:
        last_message = state["messages"][-1]
        raw_output = getattr(last_message, "content", last_message)
        # reasoning_content 兜底：qwen thinking 模式下输出可能在 reasoning 字段；
        # LangChain 不同版本会放在顶层属性或 additional_kwargs 中
        if not raw_output and self.prompt_scope is PromptScope.FORUM:
            reasoning = (
                getattr(last_message, "reasoning_content", None)
                or getattr(last_message, "additional_kwargs", {}).get("reasoning_content")
            )
            if reasoning:
                logging.info("Using reasoning_content as fallback (%d chars)", len(reasoning))
                raw_output = reasoning
        if not raw_output:
            additional = getattr(last_message, "additional_kwargs", {})
            logging.warning(
                "Final message has empty content and no reasoning. "
                "message_type=%s tool_calls=%s additional_keys=%s",
                type(last_message).__name__,
                getattr(last_message, "tool_calls", None),
                list(additional.keys()),
            )
        final_clean_text = self.parse_model_output(raw_output)
        return {
            "raw_output": raw_output,
            "final_text": final_clean_text,
        }

    async def _save_history(self, state: MentionGraphState) -> MentionGraphState:
        final_text = state.get("final_text", "")
        history_obj = state["history_obj"]
        history_obj.add_user_message(
            self._arrange_post_text(state["conversation"], state["user"])
        )
        tool_summary = self._build_tool_call_history_summary(state.get("messages", []))
        if tool_summary:
            history_obj.add_message(AIMessage(content=tool_summary))
        history_obj.add_ai_message(final_text)
        self._trim_session_history(history_obj)
        return {}

    @staticmethod
    def _arrange_post_text(raw: str, user: User) -> str:
        """
        Arrange the raw post text along with user information into a formatted string.
        Strips forum signatures before arranging to prevent the LLM from reproducing them.

        :param raw: The raw content of the post.
        :param user: The User object containing user information.
        :return: A formatted string containing the arranged post text.
        """
        # 移除签名档，避免大模型在回复中复刻签名格式
        raw = ShuiyuanModel.remove_shuiyuan_signature(raw)
        identity_info = f"- 用户【{user.username}】"
        identity_info += f" (昵称【{user.name}】)" if user.name else ""
        arranged_text = f"{identity_info}说：\n{raw}"
        return arranged_text.strip()

    async def get_recent_msgs_context(self, topic_id: int, limit: int = 10) -> str:
        """
        Get recent posts in the topic and arrange them into a text block for context.

        :param topic_id: The ID of the topic to retrieve recent posts from.
        :param limit: The maximum number of recent posts to retrieve.
        :return: A formatted string containing the recent posts.
        """
        tools_wrapper = ShuiyuanToolsWrapper(self.model)
        posts = await tools_wrapper.query_recent_posts_by_topic_id(topic_id, limit)

        # If there are no recent posts, return a default message
        if not posts:
            return "无近期回帖记录"

        # Arrange the recent posts into a formatted string
        return "\n\n".join(
            [
                self._arrange_post_text(
                    post.raw[:384], User(0, post.username, post.name)
                )
                for post in posts
            ]
        )

    @abstractmethod
    def parse_model_output(self, raw_output) -> str:
        """
        Parse the raw output from the model to extract the final response text.

        :param raw_output: The raw output from the model.
        :return: The extracted response text.
        """
        pass

    async def get_pumpkin_response(
        self,
        topic_id: Optional[int],
        reply_to_post_number: Optional[int],
        conversation: str,
        user: User,
        *,
        session_id: int | str | None = None,
        load_forum_context: bool = True,
        memory_user_id: int | str | None = None,
        external_history: tuple[ChatMessage, ...] | None = None,
        include_artifacts: bool = False,
    ) -> Optional[str]:
        """
        Let the model respond based on conversation and similar responses.

        :param topic_id: The ID of the topic where the conversation is happening.
        :param reply_to_post_number: The post number this post is replying to.
        :param conversation: The current user input or conversation snippet to respond to.
        :param user: The User object representing the user who initiated the conversation.
        :return: The model's response as a string, or None if no response is generated.
        """
        # Initialize MCP connection and LangGraph workflow if not already done.
        if self.graph is None:
            logging.info(
                "Mention graph is not initialized before request; initializing now"
            )
            await self.initialize_agent()

        logging.info(
            "Starting mention response generation: "
            "topic_id=%s reply_to_post_number=%s user=%s "
            "conversation_chars=%d conversation=%s",
            topic_id,
            reply_to_post_number,
            user.username,
            len(conversation),
            self._preview_text(conversation),
        )
        effective_session_id = topic_id if session_id is None else session_id
        if effective_session_id is None:
            raise ValueError("session_id is required when topic_id is None")
        effective_memory_user_id = user.id if memory_user_id is None else memory_user_id
        graph_input: MentionGraphState = {
            "persona": self.username,
            "topic_id": topic_id,
            "session_id": effective_session_id,
            "load_forum_context": load_forum_context,
            "memory_user_id": effective_memory_user_id,
            "reply_to_post_number": reply_to_post_number,
            "conversation": conversation,
            "user": user,
            "external_history": external_history,
        }
        memory_key = self.memory_model.memory_key(effective_memory_user_id)
        response = await self.graph.ainvoke(
            graph_input,
            config=self.memory_model.graph_config(memory_key),
        )
        final_text = response.get("final_text")

        # 空白回复重试一次
        if not final_text or not final_text.strip():
            logging.warning(
                "Empty final_text, retrying once. raw_output=%s",
                self._preview_text(response.get("raw_output"), 200),
            )
            response = await self.graph.ainvoke(
                graph_input,
                config=self.memory_model.graph_config(memory_key),
            )
            final_text = response.get("final_text")

        # 仍然空白则 fallback
        if not final_text or not final_text.strip():
            logging.warning("Still empty after retry, using fallback message.")
            final_text = "抱歉，小狼bot暂时没能生成回复，请稍后再试 :crying_cat:"

        logging.info(
            "Finished mention response generation: "
            "topic_id=%s final_chars=%d final_text=%s",
            topic_id,
            len(final_text or ""),
            self._preview_text(final_text or "", None),
        )
        if include_artifacts:
            return final_text, tuple(response.get("generated_artifacts", []) or [])
        return final_text

    async def aclose(self) -> None:
        """Release lazily initialized resources owned by the chat model."""
        await self.memory_model.aclose()
        seen: set[int] = set()
        candidates = [self.llm]
        candidates.extend(
            candidate
            for name in ("primary", "fallback")
            if (candidate := getattr(self.llm, name, None)) is not None
        )
        for candidate in candidates:
            for attribute, method_name in (
                ("http_async_client", "aclose"),
                ("http_client", "close"),
            ):
                client = getattr(candidate, attribute, None)
                if client is None or id(client) in seen:
                    continue
                seen.add(id(client))
                close = getattr(client, method_name, None)
                if close is None:
                    continue
                result = close()
                if inspect.isawaitable(result):
                    await result

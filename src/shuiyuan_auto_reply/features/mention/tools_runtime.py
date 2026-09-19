"""Tool loading (MCP + forum) and the tool-call graph nodes: log, validate, execute."""

import asyncio
import inspect
import json
import logging
import re
import uuid
from typing import Any, List, Literal, Tuple
from urllib.parse import urlparse

from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from shuiyuan_auto_reply.application.events import emit_event
from shuiyuan_auto_reply.application.retrieval_control import (
    READ_TOOLS,
    SEARCH_TOOLS,
    signature,
)
from shuiyuan_auto_reply.application.tool_results import current_turn
from shuiyuan_auto_reply.domain import GeneratedImageArtifact

from .context_budget import project_messages
from .graph_state import MentionGraphState
from .image_generation import ImageGenerationService
from .mention_multimodal import SHUIYUAN_HOSTS, ImageInspectResult
from .shuiyuan_tools_wrapper import ShuiyuanToolsWrapper
from .tool_catalog import FORUM_TOOL_NAMES


def mcp_text_content(value: Any) -> str:
    """Unwrap LangChain MCP text blocks without stringifying their envelope."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts = []
        for block in value:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif getattr(block, "type", None) == "text":
                texts.append(str(getattr(block, "text", "")))
        if texts:
            return "\n".join(texts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


class ToolsRuntimeMixin:
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
                    "sse_read_timeout": 600,
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

    def _consolidate_mcp_tools(self, tools: list[BaseTool]) -> list[BaseTool]:
        """Expose concise public-data tools; omit reply-irrelevant utilities."""
        by_name = {tool.name: tool for tool in tools}
        search = by_name.get("web_search")
        image_search = by_name.get("image_search")
        fetch = by_name.get("fetch_webpage_content")
        chuangka_menu = by_name.get("get_chuangka_menu")
        result: list[BaseTool] = []

        if search or image_search:

            async def web_search(
                query: str,
                kind: Literal["text", "news", "images"] = "text",
                max_results: int = 5,
                include_domains: list[str] | None = None,
                exclude_domains: list[str] | None = None,
            ) -> Any:
                """搜索公网网页、新闻或图片。

                何时用：问题涉及站外的事实、资讯或需要找图片素材时；水源社区内容不要
                用它，改用 forum_search。
                参数要点：kind="text" 普通网页，"news" 时效新闻，"images" 找图；
                include_domains/exclude_domains 限定站点；max_results 1–10。
                返回：items 为结果（ref/url、title、text 摘要、published_at 若有）；
                图片结果带 media 引用。摘要不等于正文，需要细节时用 web_read 读页面。
                """
                try:
                    if not query.strip():
                        raise ValueError("query must not be empty")
                    if not 1 <= max_results <= 10:
                        raise ValueError("max_results must be between 1 and 10")
                    allowed_kinds = getattr(
                        self, "_web_search_kinds", {"text", "news", "images"}
                    )
                    if kind not in allowed_kinds:
                        return {
                            "status": "error",
                            "code": "disabled_kind",
                            "message": f"web_search kind is disabled: {kind}",
                            "retryable": False,
                        }
                    target = image_search if kind == "images" else search
                    if target is None:
                        return {
                            "status": "error",
                            "code": "unsupported_kind",
                            "message": f"web_search kind is unavailable: {kind}",
                            "retryable": False,
                        }
                    args: dict[str, Any] = {
                        "query": query,
                        "max_results": max_results,
                    }
                    if kind == "news":
                        args["category"] = "news"
                    if include_domains:
                        args["include_domains"] = include_domains
                    if exclude_domains:
                        args["exclude_domains"] = exclude_domains
                    raw_value = mcp_text_content(await target.ainvoke(args))
                    try:
                        value = json.loads(raw_value)
                    except ValueError:
                        return {
                            "status": "ok",
                            "items": [{"text": raw_value[:6000]}],
                        }
                    rows = (
                        value.get("results", value.get("items", []))
                        if isinstance(value, dict)
                        else value
                    )
                    if not isinstance(rows, list):
                        rows = [rows]
                    items = []
                    for row in rows[:max_results]:
                        if not isinstance(row, dict):
                            items.append({"text": str(row)[:600]})
                            continue
                        url = row.get("url") or row.get("image")
                        item = {
                            "ref": url,
                            "url": url,
                            "title": str(row.get("title", ""))[:160],
                            "text": str(
                                row.get("snippet") or row.get("description") or ""
                            )[:500],
                            "published_at": row.get("published_at") or None,
                        }
                        if kind == "images" and url:
                            item["media"] = [
                                {
                                    "ref": f"web-image-{len(items) + 1}",
                                    "url": url,
                                }
                            ]
                        item = {
                            key: field
                            for key, field in item.items()
                            if field not in (None, "", [], {})
                        }
                        items.append(item or {"text": str(row)[:600]})
                    return {"status": "ok", "items": items}
                except Exception as exc:
                    return ShuiyuanToolsWrapper._error(exc)

            result.append(
                StructuredTool.from_function(coroutine=web_search, name="web_search")
            )

        if fetch:

            async def web_read(
                url: str = "",
                cursor: str | None = None,
                max_length: int = 8000,
                images: Literal["auto", "none"] = "none",
                mode: Literal["auto", "document", "json", "raw"] = "auto",
                query: str | None = None,
                json_path: str | None = None,
                fields: list[str] | None = None,
                max_results: int = 20,
            ) -> tuple[str, ImageInspectResult | None]:
                """读取一个公网网页并返回清洗后的正文。

                何时用：web_search 给出的页面需要看正文，或用户直接给了外站链接时。
                水源社区地址（shuiyuan.sjtu.edu.cn）不能用它，改用 forum_read。
                参数要点：query 只保留包含关键词的段落；JSON 接口可用 json_path 和
                fields 只取需要的字段；正文过长时用返回的 next_cursor 继续读。
                返回：items[0] 含 content（轻量 Markdown）、page_start，以及页面 title、
                published_at（若页面声明）；直接图片链接只返回 media 引用，images="auto"
                时才加载图片。
                """
                try:
                    if not 1 <= max_length <= 12000:
                        raise ValueError("max_length must be between 1 and 12000")
                    if not 1 <= max_results <= 100:
                        raise ValueError("max_results must be between 1 and 100")
                    offset = 0
                    if cursor:
                        state = ShuiyuanToolsWrapper._resume(cursor, "web_read")
                        if url and url != state["url"]:
                            raise ValueError(
                                "Cursor URL conflicts with the supplied URL"
                            )
                        requested = {
                            "mode": mode,
                            "query": query,
                            "json_path": json_path,
                            "fields": fields,
                            "max_results": max_results,
                        }
                        defaults = {
                            "mode": "auto",
                            "query": None,
                            "json_path": None,
                            "fields": None,
                            "max_results": 20,
                        }
                        for key, value in requested.items():
                            if value != defaults[key] and value != state[key]:
                                raise ValueError(
                                    f"Cursor extraction option conflicts: {key}"
                                )
                        url, offset = state["url"], state["offset"]
                        mode = state["mode"]
                        query = state["query"]
                        json_path = state["json_path"]
                        fields = state["fields"]
                        max_results = state["max_results"]
                    if not url.strip():
                        raise ValueError("url must not be empty")
                    if urlparse(url).netloc.lower() in SHUIYUAN_HOSTS:
                        return (
                            json.dumps(
                                {
                                    "status": "error",
                                    "code": "use_forum_tools",
                                    "message": "水源社区内容需要登录，web_read 读不到",
                                    "hint": "用 forum_read（topic_id+post_number 或 post_id）"
                                    "读取帖子；用户头像和帖内图片通过 users / forum_read 加载",
                                    "retryable": False,
                                },
                                ensure_ascii=False,
                            ),
                            None,
                        )
                    image_url = bool(
                        re.search(r"\.(?:png|jpe?g|gif|webp)(?:\?|$)", url, re.I)
                    )
                    if image_url:
                        payload = {
                            "status": "ok",
                            "items": [
                                {
                                    "ref": url,
                                    "url": url,
                                    "media": [
                                        {
                                            "ref": "image-1",
                                            "url": url,
                                            "loaded": images != "none",
                                        }
                                    ],
                                }
                            ],
                        }
                        return (
                            json.dumps(payload, ensure_ascii=False),
                            (
                                ImageInspectResult(
                                    image_urls=[url], description="网页图片"
                                )
                                if images != "none"
                                else None
                            ),
                        )
                    value = await fetch.ainvoke(
                        {
                            "url": url,
                            "max_length": max_length,
                            "start_index": offset,
                            "mode": mode,
                            "query": query,
                            "json_path": json_path,
                            "fields": fields,
                            "max_results": max_results,
                        }
                    )
                    raw_value = mcp_text_content(value)
                    try:
                        decoded = json.loads(raw_value)
                    except ValueError:
                        decoded = None
                    envelope = (
                        decoded
                        if isinstance(decoded, dict)
                        and decoded.get("status") in {"ok", "error"}
                        else None
                    )
                    if envelope and envelope.get("status") == "error":
                        return json.dumps(envelope, ensure_ascii=False), None
                    if envelope:
                        text = str(envelope.get("content", ""))
                        page_start = int(envelope.get("start_index", offset))
                        upstream_more = bool(envelope.get("truncated"))
                        next_offset = envelope.get("next_start_index")
                        source_url = str(envelope.get("url") or url)
                    else:
                        text = raw_value[:max_length]
                        page_start = offset
                        upstream_more = len(raw_value) >= max_length
                        next_offset = offset + len(text) if upstream_more else None
                        source_url = url
                    payload = {
                        "status": "ok",
                        "items": [
                            {
                                "ref": source_url,
                                "url": source_url,
                                "content": text,
                                "page_start": page_start,
                            }
                        ],
                    }
                    if envelope:
                        for key in ("title", "published_at"):
                            if envelope.get(key):
                                payload["items"][0][key] = envelope[key]
                        for key in (
                            "content_type",
                            "mode",
                            "matched_count",
                            "warnings",
                        ):
                            if envelope.get(key) not in (None, "", [], {}):
                                payload[key] = envelope[key]
                    if upstream_more:
                        payload["next_cursor"] = ShuiyuanToolsWrapper._cursor(
                            {
                                "kind": "web_read",
                                "url": url,
                                "offset": int(next_offset),
                                "mode": mode,
                                "query": query,
                                "json_path": json_path,
                                "fields": fields,
                                "max_results": max_results,
                            }
                        )
                    return json.dumps(payload, ensure_ascii=False), None
                except Exception as exc:
                    return (
                        json.dumps(
                            ShuiyuanToolsWrapper._error(exc), ensure_ascii=False
                        ),
                        None,
                    )

            result.append(
                StructuredTool.from_function(
                    coroutine=web_read,
                    name="web_read",
                    response_format="content_and_artifact",
                )
            )

        if chuangka_menu:

            async def get_chuangka_menu(
                location: Literal["all", "zhutu", "huanyuan"] = "all",
                category: Literal["all", "ice_cream"] = "all",
                query: str | None = None,
                cursor: str | None = None,
                max_length: int = 6000,
            ) -> dict[str, Any]:
                """读取交图／交环创咖当前菜单。

                何时用：用户问创咖有什么、价格或冰淇淋口味时。
                参数要点：location 选门店，category="ice_cream" 只看冰淇淋，query 按
                商品名过滤；菜单过长时用 next_cursor 继续读。
                返回：items[0].content 为菜单文本，附 total_products 与 fetched_at。
                """
                try:
                    if not 1 <= max_length <= 12000:
                        raise ValueError("max_length must be between 1 and 12000")
                    offset = 0
                    if cursor:
                        state = ShuiyuanToolsWrapper._resume(
                            cursor, "get_chuangka_menu"
                        )
                        requested = {
                            "location": location,
                            "category": category,
                            "query": query,
                        }
                        defaults = {
                            "location": "all",
                            "category": "all",
                            "query": None,
                        }
                        for key, value in requested.items():
                            if value != defaults[key] and value != state[key]:
                                raise ValueError(f"Cursor menu option conflicts: {key}")
                        location = state["location"]
                        category = state["category"]
                        query = state["query"]
                        offset = state["offset"]
                    raw_value = mcp_text_content(
                        await chuangka_menu.ainvoke(
                            {
                                "location": location,
                                "category": category,
                                "query": query,
                                "max_length": max_length,
                                "start_index": offset,
                            }
                        )
                    )
                    try:
                        envelope = json.loads(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            "MCP returned an invalid ChuangKa menu response"
                        ) from exc
                    if not isinstance(envelope, dict):
                        raise ValueError("MCP returned a non-object ChuangKa menu")
                    if envelope.get("status") == "error":
                        return envelope
                    if envelope.get("status") != "ok":
                        raise ValueError("MCP returned an unknown ChuangKa menu status")
                    source_urls = [
                        str(value) for value in envelope.get("source_urls", []) if value
                    ]
                    if not source_urls:
                        raise ValueError("ChuangKa menu response has no source URL")
                    page_start = int(envelope.get("start_index", offset))
                    content = str(envelope.get("content", ""))
                    payload: dict[str, Any] = {
                        "status": "ok",
                        "items": [
                            {
                                "ref": source_urls[0],
                                "url": source_urls[0],
                                "content": content,
                                "page_start": page_start,
                                "source_urls": source_urls,
                            }
                        ],
                    }
                    for key in (
                        "fetched_at",
                        "location",
                        "category",
                        "query",
                        "total_products",
                        "total_by_location",
                        "matched_count",
                        "failed_locations",
                        "warnings",
                    ):
                        if envelope.get(key) not in (None, "", [], {}):
                            payload[key] = envelope[key]
                    if envelope.get("truncated"):
                        next_offset = envelope.get("next_start_index")
                        if next_offset is None:
                            raise ValueError(
                                "Truncated ChuangKa menu has no next offset"
                            )
                        payload["next_cursor"] = ShuiyuanToolsWrapper._cursor(
                            {
                                "kind": "get_chuangka_menu",
                                "location": location,
                                "category": category,
                                "query": query,
                                "offset": int(next_offset),
                            }
                        )
                    return payload
                except Exception as exc:
                    return ShuiyuanToolsWrapper._error(exc)

            result.append(
                StructuredTool.from_function(
                    coroutine=get_chuangka_menu,
                    name="get_chuangka_menu",
                )
            )
        return result

    def _load_shuiyuan_tools(self) -> List[StructuredTool]:
        """Expose one model-facing tool per forum capability."""
        tools_wrapper = ShuiyuanToolsWrapper(self.model)
        tools = []
        for tool_name in FORUM_TOOL_NAMES:
            func_name = tool_name
            func = getattr(tools_wrapper, func_name)
            if callable(func):
                if tool_name == "users":

                    async def users_tool(
                        query: str | None = None,
                        username: str | None = None,
                        usernames: list[str] | None = None,
                        user_id: int | None = None,
                        include_avatar: bool = False,
                    ) -> tuple[str, ImageInspectResult | None]:
                        """Search or resolve users; optionally attach labeled avatars."""
                        payload = await tools_wrapper.users(
                            query=query,
                            username=username,
                            usernames=usernames,
                            user_id=user_id,
                            include_avatar=include_avatar,
                        )
                        urls = [
                            item["avatar"]
                            for item in payload.get("items", [])
                            if isinstance(item, dict) and item.get("avatar")
                        ]
                        artifact = (
                            ImageInspectResult(
                                image_urls=urls,
                                description="用户头像（按结果顺序）",
                            )
                            if urls
                            else None
                        )
                        return json.dumps(payload, ensure_ascii=False), artifact

                    func = users_tool
                kwargs = (
                    {"response_format": "content_and_artifact"}
                    if tool_name in {"forum_read", "users"}
                    else {}
                )
                tools.append(
                    StructuredTool.from_function(
                        coroutine=func,
                        name=tool_name,
                        description=inspect.getdoc(func)
                        or f"Tool for calling {func_name}",
                        **kwargs,
                    )
                )

        # The image tool needs the state store for artifacts; without it the
        # runtime is a text-only harness (tests, offline evaluation).
        if getattr(self, "state_store", None) is not None:
            service = ImageGenerationService(self.model, self.state_store)
            tools.append(
                StructuredTool.from_function(
                    coroutine=service.generate,
                    name="generate_image",
                    description=inspect.getdoc(service.generate),
                    response_format="content_and_artifact",
                )
            )

        logging.info(
            "Loaded %d Shuiyuan tool(s): %s",
            len(tools),
            ", ".join(tool.name for tool in tools),
        )
        return tools

    @classmethod
    async def _log_tool_calls(cls, state: MentionGraphState) -> MentionGraphState:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []

        for tool_call in tool_calls:
            tool_name, tool_args = cls._extract_tool_call_name_args(tool_call)

            logging.info(
                "Mention graph tool call: name=%s args=%s",
                tool_name,
                cls._serialize_tool_args(tool_args),
            )
            await emit_event(
                "tool.started",
                {
                    "name": tool_name,
                    "arguments": cls._prompt_event_value(tool_args),
                },
            )

        return {}

    async def _validate_tool_calls(self, state: MentionGraphState) -> MentionGraphState:
        """校验工具调用: 过滤掉幻觉的工具名和缺少必填参数的工具调用。

        对于无效调用，生成合成 ToolMessage 错误作为反馈，
        让 LLM 在下一轮知道调用失败的原因并自行纠正。
        合法调用逐项执行，错误调用也保留对应的 ToolMessage。
        """
        last_message = state["messages"][-1]
        tool_calls = list(getattr(last_message, "tool_calls", []) or [])

        if not tool_calls:
            return {}

        errors = {}
        by_name = {tool.name: tool for tool in self.tools}
        for call in tool_calls:
            name, args = self._extract_tool_call_name_args(call)
            try:
                if name not in by_name:
                    raise ValueError(f"Unknown tool: {name}")
                schema = by_name[name].args_schema
                if schema is not None and hasattr(schema, "model_validate"):
                    schema.model_validate(args)
                if name == "generate_image":
                    prompt = str(args.get("prompt", "")).strip()
                    if len(prompt) < 10 or prompt.isdigit() or len(set(prompt)) <= 2:
                        raise ValueError(
                            "generate_image requires a meaningful prompt of at least 10 characters"
                        )
            except Exception as exc:
                errors[call["id"]] = str(exc)[:500]
        return {"tool_validation_errors": errors}

    @staticmethod
    def _merge_user_lookups(calls: list) -> tuple[list, dict[str, list]]:
        """Collapse several ``users(username=…)`` calls in one batch into one lookup.

        Returns the calls to execute and a map from the merged call id to the
        original calls whose results must be split back out by ``call_id``.
        """
        singles = [
            call
            for call in calls
            if call["name"] == "users"
            and isinstance(call["args"], dict)
            and call["args"].get("username")
            and not any(
                call["args"].get(key) for key in ("query", "usernames", "user_id")
            )
        ]
        if len(singles) < 2:
            return calls, {}
        include_avatar = any(bool(c["args"].get("include_avatar")) for c in singles)
        merged = {
            "id": "merged-users:" + singles[0]["id"],
            "name": "users",
            "args": {
                "usernames": [str(c["args"]["username"]) for c in singles],
                "include_avatar": include_avatar,
            },
            "type": "tool_call",
        }
        single_ids = {c["id"] for c in singles}
        rest = [call for call in calls if call["id"] not in single_ids]
        return rest + [merged], {merged["id"]: singles}

    @staticmethod
    def _split_user_lookup(message: ToolMessage, originals: list) -> list:
        """Rebuild one ToolMessage per original call from a batched users result."""
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError):
            payload = None
        items = payload.get("items", []) if isinstance(payload, dict) else []
        by_input = {}
        for item in items:
            if isinstance(item, dict) and item.get("input"):
                by_input[str(item["input"]).strip().lstrip("@").casefold()] = item
        results = []
        for call in originals:
            key = str(call["args"]["username"]).strip().lstrip("@").casefold()
            item = by_input.get(key)
            if item is None or item.get("status") != "ok":
                content = {
                    "status": "error",
                    "code": (item or {}).get("code", "not_found"),
                    "message": (item or {}).get("message", "User was not found"),
                    "retryable": False,
                }
                status = "error"
            else:
                clean = {
                    k: v
                    for k, v in item.items()
                    if k not in {"input", "status"} and v not in (None, "", [], {})
                }
                content = {"status": "ok", "items": [clean]}
                status = "success"
            results.append(
                ToolMessage(
                    content=json.dumps(content, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name="users",
                    status=status,
                    artifact=getattr(message, "artifact", None),
                )
            )
        return results

    async def _execute_tools(self, state: MentionGraphState):
        original_calls = state["messages"][-1].tool_calls
        errors = state.get("tool_validation_errors", {})
        clean_calls = [c for c in original_calls if c["id"] not in errors]
        calls, merged_users = ToolsRuntimeMixin._merge_user_lookups(clean_calls)
        calls += [c for c in original_calls if c["id"] in errors]
        by_name = {tool.name: tool for tool in self.tools}
        turn = current_turn.get()
        prior_pages = len(turn.read_pages) if turn else 0
        pending_signatures = set()
        prepared = {}
        for call in calls:
            name, args = call["name"], dict(call["args"])
            error = errors.get(call["id"])
            cached = None
            if turn and not error:
                control, progress = turn.control, turn.progress
                sig = signature(name, args)
                if (
                    name in READ_TOOLS
                    and sig in control.seen
                    and not args.get("refresh")
                ):
                    control.repeats += 1
                    cached = control.seen[sig]
                elif name in READ_TOOLS and sig in pending_signatures:
                    error = "Duplicate read in this batch; reuse its result"
                    control.repeats += 1
                elif progress.phase == "final" and name in READ_TOOLS:
                    error = "Read budget finished; answer from the available results"
                elif name in READ_TOOLS:
                    if control.queries >= control.query_limit:
                        control.stop(progress, "query_budget")
                        error = (
                            "Read query budget reached; answer from existing evidence"
                        )
                    elif not error:
                        control.queries += 1
                pending_signatures.add(sig)
            prepared[call["id"]] = (args, error, cached)

        async def execute(call):
            import time

            started_at = time.monotonic()
            args, error, cached = prepared[call["id"]]
            if cached is not None:
                # Replaying a stored result must look like a new message: graphs merge
                # the message list by id, so reusing the stored id would replace the
                # earlier message in place and leave this call without a result.
                return cached.model_copy(
                    update={"tool_call_id": call["id"], "id": str(uuid.uuid4())}
                )
            if error:
                return ToolMessage(
                    content=error,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            call = {**call, "args": args}
            if call["id"] in errors:
                return ToolMessage(
                    content=errors[call["id"]],
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            try:
                if turn:
                    await emit_event(
                        "tool.execution", {"name": call["name"], "arguments": args}
                    )
                message = await by_name[call["name"]].ainvoke(
                    {**call, "type": "tool_call"}
                )
                await emit_event(
                    "tool.timing",
                    {
                        "name": call["name"],
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    },
                )
                try:
                    payload = (
                        json.loads(message.content)
                        if isinstance(message.content, str)
                        else None
                    )
                except (ValueError, TypeError):
                    payload = None
                if isinstance(payload, dict) and payload.get("status") == "error":
                    message = message.model_copy(update={"status": "error"})
                return message
            except Exception as exc:
                await emit_event(
                    "tool.timing",
                    {
                        "name": call["name"],
                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                    },
                )
                return ToolMessage(
                    content=str(exc)[:500],
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )

        try:
            if turn:
                import time

                async with asyncio.timeout(
                    turn.control.call_timeout(turn.deadline, final=False)
                ):
                    responses = await asyncio.gather(*(execute(call) for call in calls))
            else:
                responses = await asyncio.gather(*(execute(call) for call in calls))
        except TimeoutError:
            turn.control.stop(turn.progress, "tool_time_budget")
            responses = [
                ToolMessage(
                    content="Tool batch exceeded investigation deadline; stop and answer",
                    tool_call_id=c["id"],
                    name=c["name"],
                    status="error",
                )
                for c in calls
            ]
        if merged_users:
            expanded_calls, expanded_responses = [], []
            for call, message in zip(calls, responses):
                originals = merged_users.get(call["id"])
                if originals is None:
                    expanded_calls.append(call)
                    expanded_responses.append(message)
                    continue
                split = ToolsRuntimeMixin._split_user_lookup(message, originals)
                expanded_calls.extend(originals)
                expanded_responses.extend(split)
            order = {c["id"]: i for i, c in enumerate(original_calls)}
            paired = sorted(
                zip(expanded_calls, expanded_responses),
                key=lambda pair: order.get(pair[0]["id"], len(order)),
            )
            calls = [c for c, _ in paired]
            responses = [m for _, m in paired]
        seen_errors = {}
        for index, (call, message) in enumerate(zip(calls, responses)):
            if message.status != "error":
                continue
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("retryable") is not False:
                continue
            key = (call["name"], payload.get("code"), payload.get("message"))
            if key not in seen_errors:
                seen_errors[key] = True
                continue
            responses[index] = message.model_copy(
                update={
                    "content": json.dumps(
                        {
                            "status": "error",
                            "code": "duplicate_error",
                            "message": (
                                "Same non-retryable error as an earlier call; "
                                "correct it once"
                            ),
                            "retryable": False,
                        },
                        ensure_ascii=False,
                    )
                }
            )
        turn = current_turn.get()
        if turn:
            added = set()
            for call, message in zip(calls, responses):
                args, error, cached = prepared[call["id"]]
                if not error and cached is None:
                    added.update(turn.observe(message.content, tool=call["name"]))
                    if call["name"] in READ_TOOLS:
                        turn.control.seen[signature(call["name"], args)] = message
                if call["name"] in SEARCH_TOOLS:
                    turn.progress.searches.append(
                        {
                            "tool": call["name"],
                            "args": args,
                            "failed": message.status == "error",
                        }
                    )
                    turn.progress.searches = turn.progress.searches[-40:]
                turn.save(
                    {
                        "tool": call["name"],
                        "args": call["args"],
                        "status": message.status,
                        "output": message.content,
                    }
                )
            turn.control.after_batch(
                turn.progress,
                new_evidence=len(added) + len(turn.read_pages) - prior_pages,
                reads=sum(c["name"] in READ_TOOLS for c in calls),
            )
            read_calls = [c for c in calls if c["name"] in READ_TOOLS]
            if (
                read_calls
                and not added
                and len(turn.read_pages) == prior_pages
                and all(
                    turn.is_redundant_completed_call(c["name"], c["args"])
                    for c in read_calls
                )
            ):
                turn.control.stop(turn.progress, "source_complete")
            await emit_event(
                "retrieval.batch",
                {"new_evidence": len(added), **turn.control.metrics()},
            )
        return {"messages": responses, "tool_validation_errors": {}}

    def _has_valid_tool_calls(self, state: MentionGraphState) -> str:
        """条件路由: 验证后是否还有合法工具调用需要执行。

        返回 "tools" → 执行合法调用并为无效调用生成错误响应
        返回 "call_model" → 所有调用都被过滤了, 让 LLM 看到错误并纠正
        """
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []
        return "tools" if tool_calls else "call_model"

    @classmethod
    async def _log_tool_outputs(cls, state: MentionGraphState) -> MentionGraphState:
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
                cls._preview_text(getattr(message, "content", message)),
            )
            event_type = (
                "tool.failed"
                if getattr(message, "status", None) == "error"
                else "tool.completed"
            )
            await emit_event(
                event_type,
                {
                    "name": getattr(message, "name", "<unknown>"),
                    "output": cls._preview_text(
                        getattr(message, "content", message), 2000
                    ),
                },
            )
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, GeneratedImageArtifact):
                generated.append(artifact)
                await emit_event(
                    "image.generated",
                    {
                        "artifact_id": artifact.artifact_id,
                        "byte_count": artifact.byte_count,
                    },
                )

        return {"generated_artifacts": generated}

    async def _collect_tool_output_images(
        self, state: MentionGraphState
    ) -> MentionGraphState:
        return {"image_inputs": list(state.get("image_inputs", []) or [])}

    @staticmethod
    def _trim_tool_loop_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
        from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

        budget = int(get_deployment().section("runtime")["context_token_budget"])
        return project_messages(messages, budget)

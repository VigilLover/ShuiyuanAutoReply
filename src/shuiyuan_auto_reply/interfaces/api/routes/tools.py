"""Built-in tool allowlist and MCP server status/catalog for the settings page."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from shuiyuan_auto_reply.features.mention.mention_chat_model import MentionChatModel
from shuiyuan_auto_reply.features.mention.tool_catalog import (
    MODEL_TOOL_NAMES,
    migrate_tool_names,
)

from ..support import profile_defaults, state_store

logger = logging.getLogger(__name__)
router = APIRouter()

# Tool names the agent registers, used as the settings-page fallback before the
# first agent run writes its own catalog. Keep in sync with the registration in
# MentionChatModel._load_shuiyuan_tools.
RUNTIME_TOOL_NAMES = MODEL_TOOL_NAMES


@router.get("/api/settings/tools/{scope}")
async def get_tools(scope: str, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    catalog = await state_store(request).list_tool_catalog(scope)
    if catalog:
        catalog = [item for item in catalog if item.get("source") != "mcp"]
        profile = await state_store(request).get_profile(scope, profile_defaults(scope))
        configured = migrate_tool_names(profile["draft"].get("enabled_tools"))
        if configured is not None:
            selected = set(configured)
            for item in catalog:
                item["enabled"] = item["name"] in selected
        return catalog
    names = list(RUNTIME_TOOL_NAMES)
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    configured = migrate_tool_names(profile["draft"].get("enabled_tools"))
    enabled = set(configured) if configured is not None else set(names)
    return [
        {"name": name, "enabled": name in enabled, "source": "runtime"}
        for name in names
    ]


@router.get("/api/settings/mcp/{scope}")
async def get_mcp_status(scope: str, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    container = request.app.state.container
    url = container.settings.providers.mcp_server_url
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    disabled = set(profile["draft"].get("disabled_mcp_tools", []))
    if not url:
        return {
            "url": None,
            "configured": False,
            "connected": False,
            "error": "MCP_SERVER_URL 未配置",
            "tools": [],
        }
    try:
        loaded_tools = await asyncio.wait_for(
            MentionChatModel._load_mcp_tools(url), timeout=15
        )
    except Exception as exc:
        logger.warning("MCP status probe failed for %s: %s", url, exc)
        return {
            "url": url,
            "configured": True,
            "connected": False,
            "error": str(exc)[:300],
            "tools": [],
        }
    loaded_names = {tool.name for tool in loaded_tools}
    tools = []
    if loaded_names & {"web_search", "image_search"}:
        allowed_kinds = set()
        if "web_search" in loaded_names and "web_search" not in disabled:
            allowed_kinds.update({"text", "news"})
        if "image_search" in loaded_names and "image_search" not in disabled:
            allowed_kinds.add("images")
        tools.append(
            {
                "name": "web_search",
                "description": "统一网页、新闻和图片搜索",
                "enabled": bool(allowed_kinds),
                "kinds": sorted(allowed_kinds),
            }
        )
    if "fetch_webpage_content" in loaded_names:
        tools.append(
            {
                "name": "web_read",
                "description": "读取网页正文或直接图片",
                "enabled": "fetch_webpage_content" not in disabled,
            }
        )
    if "get_chuangka_menu" in loaded_names:
        tools.append(
            {
                "name": "get_chuangka_menu",
                "description": "读取交图、交环创咖当前菜单或冰淇淋菜单",
                "enabled": "get_chuangka_menu" not in disabled,
            }
        )
    return {
        "url": url,
        "configured": True,
        "connected": True,
        "error": None,
        "tools": tools,
    }

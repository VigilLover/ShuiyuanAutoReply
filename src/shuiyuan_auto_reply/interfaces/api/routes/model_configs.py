"""The stored chat/image endpoint library: CRUD, connectivity probe, activation."""

import asyncio
import logging
import os
from typing import Any, Literal

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from shuiyuan_auto_reply.bootstrap import AppSettings
from shuiyuan_auto_reply.infrastructure.persistence.model_configs import (
    secret_name as model_config_secret_name,
)

from ..support import (
    DEEPSEEK_VISION_MODEL,
    normalized_base_url,
    profile_defaults,
    state_store,
)
from .settings_profiles import apply_scope_profile

logger = logging.getLogger(__name__)
router = APIRouter()

PROBE_TIMEOUT_SECONDS = 10.0


class ModelConfigRequest(BaseModel):
    """One stored endpoint configuration for the chat or image model."""

    kind: Literal["chat", "image"]
    name: str
    base_url: str
    model: str
    api_format: Literal["chat_completions", "responses"] | None = None
    api_key: str | None = None


class ModelConfigProbeRequest(BaseModel):
    kind: Literal["chat", "image"]
    base_url: str
    api_key: str | None = None
    model: str | None = None
    config_id: str | None = None


class ModelConfigActivateRequest(BaseModel):
    scope: Literal["web", "forum", "image"]


async def list_provider_models(base_url: str, api_key: str | None) -> list[str]:
    """Read {base_url}/models; the only reliable "is this endpoint usable" probe."""
    url = normalized_base_url(base_url)
    if not url:
        raise ValueError("Base URL 为空")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = aiohttp.ClientTimeout(total=PROBE_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{url}/models", headers=headers) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            payload = await response.json(content_type=None)
    identifiers = [
        str(item["id"])
        for item in (payload.get("data") or [])
        if isinstance(item, dict) and item.get("id")
    ]
    if not identifiers:
        raise ValueError("响应里没有模型列表")
    return sorted(set(identifiers))


async def decorated_model_config(request: Request, config: dict) -> dict:
    """Add the key status a configuration entry is shown with."""
    entry = dict(config)
    vault = request.app.state.container.secret_vault
    if entry.get("source") == "default":
        env_name = (
            "IMAGE_GEN_API_KEY" if entry["kind"] == "image" else "DEEPSEEK_API_KEY"
        )
        value = os.getenv(env_name)
        entry["secret"] = {
            "configured": bool(value),
            "last_four": value[-4:] if value else None,
            "source": "environment" if value else None,
        }
        return entry
    stored = ""
    if vault is not None:
        stored = (await vault.get(model_config_secret_name(entry["id"])) or "").strip()
    entry["secret"] = {
        "configured": bool(stored),
        "last_four": stored[-4:] if stored else None,
        "source": "ui" if stored else None,
    }
    return entry


def default_model_config(kind: str) -> dict:
    """The built-in endpoint: deployment configuration, never editable."""
    if kind == "image":
        return {
            "id": "default",
            "kind": "image",
            "name": "默认（部署配置）",
            "base_url": os.getenv("IMAGE_GEN_API_URL", "").strip(),
            "model": os.getenv("IMAGE_GEN_MODEL", "").strip() or "gpt-image-2",
            "api_format": None,
            "source": "default",
        }
    return {
        "id": "default",
        "kind": "chat",
        "name": "默认（官方 DeepSeek）",
        "base_url": "",
        "model": DEEPSEEK_VISION_MODEL,
        "api_format": AppSettings().providers.deepseek_api_format.value,
        "source": "default",
    }


def validated_model_config(payload: ModelConfigRequest) -> dict:
    name = payload.name.strip()
    model = payload.model.strip()
    if not name:
        raise HTTPException(status_code=400, detail="配置名称不能为空")
    if not model:
        raise HTTPException(status_code=400, detail="模型名称不能为空")
    return {
        "kind": payload.kind,
        "name": name[:60],
        "model": model,
        "base_url": normalized_base_url(payload.base_url),
        "api_format": payload.api_format if payload.kind == "chat" else None,
    }


async def store_model_config_secret(
    request: Request, config_id: str, api_key: str | None
) -> None:
    """An empty or absent key keeps whatever the entry already had."""
    vault = request.app.state.container.secret_vault
    if vault is None or not api_key:
        return
    await vault.set(model_config_secret_name(config_id), api_key.strip())


@router.get("/api/settings/model-configs")
async def get_model_configs(request: Request):
    store = state_store(request)
    active: dict[str, str] = {}
    for scope in ("web", "forum", "image"):
        row = await store.active_model_config(scope)
        active[scope] = row["id"] if row else "default"
    result: dict[str, Any] = {"active": active}
    for kind in ("chat", "image"):
        entries = [await decorated_model_config(request, default_model_config(kind))]
        for stored in await store.list_model_configs(kind):
            entries.append(
                await decorated_model_config(request, {**stored, "source": "custom"})
            )
        result[kind] = entries
    return result


@router.post("/api/settings/model-configs")
async def create_model_config(payload: ModelConfigRequest, request: Request):
    value = validated_model_config(payload)
    config_id = await state_store(request).save_model_config(value)
    await store_model_config_secret(request, config_id, payload.api_key)
    return {"status": "created", "id": config_id}


@router.put("/api/settings/model-configs/{config_id}")
async def update_model_config(
    config_id: str, payload: ModelConfigRequest, request: Request
):
    existing = await state_store(request).model_config(config_id)
    if existing is None or existing["kind"] != payload.kind:
        raise HTTPException(status_code=404, detail="未知的模型配置")
    value = validated_model_config(payload)
    try:
        await state_store(request).save_model_config(value, config_id=config_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="未知的模型配置") from exc
    await store_model_config_secret(request, config_id, payload.api_key)
    return {"status": "updated", "id": config_id}


@router.delete("/api/settings/model-configs/{config_id}")
async def delete_model_config(config_id: str, request: Request):
    store = state_store(request)
    if await store.model_config(config_id) is None:
        raise HTTPException(status_code=404, detail="未知的模型配置")
    await store.delete_model_config(config_id)
    vault = request.app.state.container.secret_vault
    if vault is not None:
        await vault.set(model_config_secret_name(config_id), "")
    return {"status": "deleted"}


@router.post("/api/settings/model-configs/probe")
async def probe_model_config(payload: ModelConfigProbeRequest, request: Request):
    """Check that an endpoint answers and that the key is accepted."""
    vault = request.app.state.container.secret_vault
    api_key = (payload.api_key or "").strip()
    if not api_key and payload.config_id and vault is not None:
        api_key = (
            await vault.get(model_config_secret_name(payload.config_id)) or ""
        ).strip()
    if not api_key:
        env_name = (
            "IMAGE_GEN_API_KEY" if payload.kind == "image" else "DEEPSEEK_API_KEY"
        )
        api_key = (os.getenv(env_name) or "").strip()
    try:
        models = await list_provider_models(payload.base_url, api_key or None)
    except HTTPException as exc:
        return {
            "ok": False,
            "models": [],
            "model_present": False,
            "message": f"探测失败：{str(exc.detail)[:300]}",
        }
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        return {
            "ok": False,
            "models": [],
            "model_present": False,
            "message": f"探测失败：{(str(exc) or type(exc).__name__)[:300]}",
        }
    wanted = (payload.model or "").strip()
    present = bool(wanted) and wanted in models
    if not wanted:
        message = f"连接成功，返回 {len(models)} 个模型"
    elif present:
        message = f"连接成功，{wanted} 在模型列表中"
    else:
        message = f"连接成功，但列表里没有 {wanted}，仍可手动填写"
    return {
        "ok": True,
        "models": models,
        "model_present": present,
        "message": message,
    }


@router.post("/api/settings/model-configs/{config_id}/activate")
async def activate_model_config(
    config_id: str, payload: ModelConfigActivateRequest, request: Request
):
    """Switch a scope onto this configuration; chat goes through the hot swap."""
    store = state_store(request)
    scope = payload.scope
    if scope == "image":
        if config_id == "default":
            await store.clear_active_model_config("image")
            return {"status": "active", "scope": scope, "config_id": "default"}
        config = await store.model_config(config_id)
        if config is None or config["kind"] != "image":
            raise HTTPException(status_code=404, detail="未知的生图配置")
        await store.set_active_model_config("image", config_id)
        return {"status": "active", "scope": scope, "config_id": config_id}
    defaults = profile_defaults(scope)
    profile = await store.get_profile(scope, defaults)
    draft = dict(profile["draft"])
    if config_id == "default":
        draft["base_url"] = ""
        draft["model"] = defaults["model"]
    else:
        config = await store.model_config(config_id)
        if config is None or config["kind"] != "chat":
            raise HTTPException(status_code=404, detail="未知的文字模型配置")
        draft["base_url"] = config["base_url"]
        draft["model"] = config["model"]
        if config.get("api_format"):
            draft["api_format"] = config["api_format"]
    await store.save_profile_draft(scope, draft)
    revision = await apply_scope_profile(scope, request)
    if config_id == "default":
        await store.clear_active_model_config(scope)
    else:
        await store.set_active_model_config(scope, config_id)
    return {
        "status": "active",
        "scope": scope,
        "config_id": config_id,
        "active_revision": revision,
    }

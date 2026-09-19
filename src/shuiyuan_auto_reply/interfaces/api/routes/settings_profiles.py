"""Runtime profile (model + prompt + tool allowlist) CRUD and hot-swap."""

import asyncio
import logging
import os
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from shuiyuan_auto_reply.bootstrap import AppSettings
from shuiyuan_auto_reply.bootstrap.settings import DeepSeekApiFormat
from shuiyuan_auto_reply.features.mention.tool_catalog import migrate_tool_names

from ..support import (
    DEEPSEEK_VISION_MODEL,
    normalized_base_url,
    profile_defaults,
    state_store,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class ProfileDraftRequest(BaseModel):
    provider: str
    model: str | None = None
    api_format: DeepSeekApiFormat = Field(
        default_factory=lambda: AppSettings().providers.deepseek_api_format
    )
    fallback_model: str | None = None
    system_prompt: str = ""
    prompt_mode: Literal["managed", "legacy"] = "legacy"
    persona_id: str = "wolf_lumine"
    persona_text: str | None = None
    additional_instructions: str = ""
    enabled_tools: list[str] | None = None
    disabled_mcp_tools: list[str] = Field(default_factory=list)
    api_key: str | None = None
    base_url: str | None = None


@router.get("/api/settings/profiles")
async def get_profiles(request: Request):
    store = state_store(request)
    profiles = [
        await store.get_profile(scope, profile_defaults(scope))
        for scope in ("forum", "web")
    ]
    from shuiyuan_auto_reply.infrastructure.prompts.profiles import profile_metadata

    for profile in profiles:
        active = {k: v for k, v in profile["active"].items() if k != "profile_revision"}
        profile["draft_changed"] = active != profile["draft"]
        profile["prompt_metadata"] = profile_metadata(active, profile["scope"])
        configured = migrate_tool_names(profile["draft"].get("enabled_tools"))
        profile["suggested_tools"] = [
            name
            for name in ("forum_search", "forum_read", "users")
            if configured is not None and name not in configured
        ]
    vault = request.app.state.container.secret_vault
    for profile in profiles:
        for value in (profile["draft"], profile["active"]):
            value["provider"] = "deepseek"
            if not value.get("model"):
                value["model"] = DEEPSEEK_VISION_MODEL
            value["base_url"] = normalized_base_url(value.get("base_url"))
            value.setdefault(
                "api_format", AppSettings().providers.deepseek_api_format.value
            )
            value["fallback_model"] = None
        provider = "deepseek"
        metadata = (
            await vault.metadata(f"{profile['scope']}:{provider}")
            if vault
            else {"configured": False}
        )
        if metadata.get("configured"):
            metadata["source"] = "ui"
        else:
            environment_value = os.getenv("DEEPSEEK_API_KEY")
            metadata.update(
                {
                    "configured": bool(environment_value),
                    "source": "environment" if environment_value else None,
                    "last_four": (
                        environment_value[-4:] if environment_value else None
                    ),
                }
            )
        profile["secret"] = metadata
    return profiles


@router.put("/api/settings/profiles/{scope}/draft")
async def save_profile(scope: str, payload: ProfileDraftRequest, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    if payload.provider != "deepseek":
        raise HTTPException(
            status_code=400, detail="视觉流程固定使用 DeepSeek 兼容客户端"
        )
    model = (payload.model or DEEPSEEK_VISION_MODEL).strip()
    if not model:
        raise HTTPException(status_code=400, detail="模型名称不能为空")
    value = payload.model_dump(mode="json", exclude={"api_key"})
    value["model"] = model
    value["base_url"] = normalized_base_url(payload.base_url)
    value["fallback_model"] = None
    await state_store(request).get_profile(scope, profile_defaults(scope))
    await state_store(request).save_profile_draft(scope, value)
    if payload.api_key:
        await request.app.state.container.secret_vault.set(
            f"{scope}:{payload.provider}", payload.api_key
        )
    return {"status": "saved"}


@router.post("/api/settings/profiles/{scope}/validate")
async def validate_profile(scope: str, request: Request):
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    draft = profile["draft"]
    errors = []
    if not draft.get("provider"):
        errors.append("Provider 不能为空")
    if (
        draft.get("prompt_mode") != "managed"
        and not str(draft.get("system_prompt", "")).strip()
    ):
        errors.append("System Prompt 不能为空")
    return {"valid": not errors, "errors": errors}


async def apply_scope_profile(scope: str, request: Request) -> int:
    """Build a candidate runtime for the saved draft, then make it active."""
    validation = await validate_profile(scope, request)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["errors"])
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    container = request.app.state.container
    prepared = None
    forum_candidate = None
    profile["draft"]["profile_revision"] = profile["active_revision"] + 1
    prepare_runtime = getattr(container, "prepare_runtime_profile", None)
    if scope == "web" and prepare_runtime is not None:
        try:
            prepared = await prepare_runtime(scope, profile["draft"])
        except Exception as exc:
            logger.exception("候选 Web Runtime 构建失败")
            raise HTTPException(
                status_code=400, detail=f"Runtime 构建失败: {exc}"
            ) from exc
    prepare_forum = getattr(container, "prepare_forum_runtime_profile", None)
    if scope == "forum" and prepare_forum is not None:
        try:
            forum_candidate = await prepare_forum(profile["draft"])
        except Exception as exc:
            logger.exception("候选 Forum Runtime 构建失败")
            raise HTTPException(
                status_code=400, detail=f"Runtime 构建失败: {exc}"
            ) from exc
    try:
        revision = await state_store(request).apply_profile(scope)
    except Exception:
        if prepared is not None:
            await prepared[0].aclose()
        if forum_candidate is not None:
            await forum_candidate.aclose()
        raise
    if forum_candidate is not None:
        await forum_candidate.aclose()
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    if prepared is not None:
        await container.activate_prepared_runtime(prepared)
    else:
        apply_runtime = getattr(container, "apply_runtime_profile", None)
        if apply_runtime is not None and scope == "web":
            await apply_runtime(scope, profile["active"])
    return revision


@router.post("/api/settings/profiles/{scope}/apply")
async def apply_profile(scope: str, request: Request):
    return {
        "status": "applied",
        "active_revision": await apply_scope_profile(scope, request),
    }


@router.post("/api/settings/profiles/{scope}/provider-test")
async def provider_test(scope: str, request: Request):
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    provider = profile["draft"].get("provider", "deepseek")
    secret = await request.app.state.container.secret_vault.get(f"{scope}:{provider}")
    if not (secret or os.getenv("DEEPSEEK_API_KEY")):
        return {"ok": False, "message": "缺少 API Key"}
    candidate = None
    handler = None
    try:
        container = request.app.state.container
        if scope == "web" and hasattr(container, "prepare_runtime_profile"):
            handler, _service = await container.prepare_runtime_profile(
                scope, profile["draft"]
            )
            candidate = getattr(getattr(handler, "_backend", None), "model", None)
        elif scope == "forum" and hasattr(container, "prepare_forum_runtime_profile"):
            candidate = await container.prepare_forum_runtime_profile(profile["draft"])
        if candidate is not None:
            await asyncio.wait_for(
                candidate.llm.ainvoke("Reply with exactly: OK"), timeout=45
            )
        return {"ok": True, "message": "Provider 连接测试成功"}
    except Exception as exc:
        logger.exception("Provider connection test failed")
        return {"ok": False, "message": f"Provider 连接失败: {str(exc)[:300]}"}
    finally:
        if handler is not None:
            await handler.aclose()
        elif candidate is not None:
            await candidate.aclose()


@router.post("/api/settings/profiles/{scope}/prompt-preview")
async def preview_prompt(scope: str, payload: ProfileDraftRequest):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    from shuiyuan_auto_reply.infrastructure.prompts.profiles import (
        profile_metadata,
        render_profile,
    )

    value = payload.model_dump(exclude={"api_key"})
    return {
        "template": render_profile(value, scope),
        **profile_metadata(value, scope),
    }


@router.post("/api/settings/profiles/{scope}/prompt-migrate")
async def migrate_prompt(scope: str, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    draft = profile["draft"]
    draft.update(prompt_mode="managed", persona_text=None, additional_instructions="")
    await state_store(request).save_profile_draft(scope, draft)
    return {
        "status": "draft_updated",
        "message": "旧完整提示词已保留；请审阅人设并应用",
    }


@router.post("/api/settings/profiles/{scope}/restore-persona")
async def restore_persona(scope: str, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    profile = await state_store(request).get_profile(scope, profile_defaults(scope))
    draft = profile["draft"]
    draft["persona_text"] = None
    await state_store(request).save_profile_draft(scope, draft)
    return {"status": "draft_updated"}


@router.post("/api/settings/profiles/{scope}/restore-default")
async def restore_profile_default(scope: str, request: Request):
    if scope not in {"forum", "web"}:
        raise HTTPException(status_code=404, detail="未知应用")
    defaults = profile_defaults(scope)
    await state_store(request).get_profile(scope, defaults)
    await state_store(request).save_profile_draft(scope, defaults)
    # The library entry no longer describes the draft, so stop claiming it.
    await state_store(request).clear_active_model_config(scope)
    return {"status": "restored"}

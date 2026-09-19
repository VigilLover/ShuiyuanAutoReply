"""Formal forum worker entry point."""

import asyncio
import logging
from dataclasses import replace

from dotenv import load_dotenv

from shuiyuan_auto_reply.application.handlers import ChatHandler
from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.bootstrap.container import ApplicationContainer
from shuiyuan_auto_reply.bootstrap.providers import (
    MentionProviderFactory,
    apply_profile_endpoint,
)
from shuiyuan_auto_reply.bootstrap.settings import AppSettings, DeepSeekApiFormat
from shuiyuan_auto_reply.features.mention import MentionModel
from shuiyuan_auto_reply.infrastructure.llm import LegacyMentionChatBackend
from shuiyuan_auto_reply.infrastructure.persistence import (
    LocalSecretVault,
    SQLiteStateStore,
)
from shuiyuan_auto_reply.infrastructure.persistence.model_configs import (
    image_endpoint_resolver,
)
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

DEEPSEEK_VISION_MODEL = "deepseek-flash"


def _forum_profile_defaults(settings: AppSettings, persona: str) -> dict:
    prompt = (
        FilePromptRepository().load(persona, set(), PromptScope.FORUM).system_prompt
    )
    return {
        "provider": "deepseek",
        "model": DEEPSEEK_VISION_MODEL,
        "base_url": "",
        "api_format": settings.providers.deepseek_api_format.value,
        "fallback_model": None,
        "system_prompt": prompt,
        "enabled_tools": None,
        "disabled_mcp_tools": [],
    }


async def _forum_provider_settings(
    settings: AppSettings, store, vault: LocalSecretVault, profile: dict
):
    provider = "deepseek"
    effective = replace(
        settings.providers,
        mention_provider=provider,
        deepseek_model=DEEPSEEK_VISION_MODEL,
        deepseek_api_format=DeepSeekApiFormat(
            profile.get("api_format", settings.providers.deepseek_api_format.value)
        ),
    )
    secret = await vault.get(f"forum:{provider}")
    if secret:
        effective = replace(effective, deepseek_api_key=secret)
    return await apply_profile_endpoint(
        effective, "forum", profile, store=store, vault=vault
    )


async def run_worker(persona: str = "wolf_lumine") -> None:
    load_dotenv()
    settings = AppSettings()
    state_store = SQLiteStateStore()
    await state_store.initialize()
    secret_vault = LocalSecretVault(state_store)
    # The image tool only receives the store, so the resolver travels with it.
    state_store.model_config_resolver = image_endpoint_resolver(
        state_store, secret_vault
    )
    model = await ShuiyuanModel.create(settings.forum.cookie_file)
    from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

    if get_deployment().profile == "remote":
        try:
            await model.verify_identity(settings.forum.bot_username)
            from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
                check_vector_space,
                engine_for,
            )

            engine = engine_for("memory_url")
            try:
                async with engine.connect() as connection:
                    await check_vector_space(connection)
            finally:
                await engine.dispose()
        except BaseException:
            await model.close()
            raise
    chat_model = None
    mention = None
    container = None
    try:
        profile = await state_store.get_profile(
            "forum", _forum_profile_defaults(settings, persona)
        )
        effective_settings = await _forum_provider_settings(
            settings, state_store, secret_vault, profile["active"]
        )
        chat_model = MentionProviderFactory.create(
            model,
            persona,
            effective_settings,
            prompt_scope=PromptScope.FORUM,
            enabled_tools=(
                set(profile["active"]["enabled_tools"])
                if profile["active"].get("enabled_tools") is not None
                else None
            ),
            disabled_mcp_tools=set(profile["active"].get("disabled_mcp_tools", [])),
            state_store=state_store,
            prompt_profile=profile["active"],
        )
        active_revision = profile["active_revision"]
        refresh_lock = asyncio.Lock()

        async def refresh_runtime() -> None:
            nonlocal active_revision
            async with refresh_lock:
                latest = await state_store.get_profile(
                    "forum", _forum_profile_defaults(settings, persona)
                )
                if latest["active_revision"] == active_revision:
                    return
                candidate_settings = await _forum_provider_settings(
                    settings, state_store, secret_vault, latest["active"]
                )
                enabled = latest["active"].get("enabled_tools")
                candidate = MentionProviderFactory.create(
                    model,
                    persona,
                    candidate_settings,
                    prompt_scope=PromptScope.FORUM,
                    enabled_tools=set(enabled) if enabled is not None else None,
                    disabled_mcp_tools=set(
                        latest["active"].get("disabled_mcp_tools", [])
                    ),
                    state_store=state_store,
                    prompt_profile=latest["active"],
                )
                await mention.swap_chat_model(candidate)
                active_revision = latest["active_revision"]
                logging.info(
                    "Applied forum runtime revision %s (previous runtime retires at shutdown)",
                    active_revision,
                )

        mention = MentionModel(
            model,
            bot_username=settings.forum.bot_username,
            persona=persona,
            chat_model=chat_model,
            provider_settings=effective_settings,
            state_store=state_store,
            runtime_refresher=refresh_runtime,
        )
        container = ApplicationContainer(
            settings,
            model,
            mention.bot_service,
            ChatHandler(LegacyMentionChatBackend(chat_model, owns_model=False)),
            managed=[mention],
            state_store=state_store,
            secret_vault=secret_vault,
        )
        await mention.watch_new_action_routine()
    finally:
        if container is not None:
            await container.aclose()
        else:
            try:
                if chat_model is not None:
                    await chat_model.aclose()
            finally:
                await model.close()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

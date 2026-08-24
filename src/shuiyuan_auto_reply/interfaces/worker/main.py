"""Formal forum worker entry point."""

import asyncio
import logging
from dataclasses import replace

from dotenv import load_dotenv

from shuiyuan_auto_reply.application.handlers import ChatHandler
from shuiyuan_auto_reply.bootstrap.container import ApplicationContainer
from shuiyuan_auto_reply.bootstrap.providers import MentionProviderFactory
from shuiyuan_auto_reply.bootstrap.settings import AppSettings, DeepSeekApiFormat
from shuiyuan_auto_reply.features.mention import MentionModel
from shuiyuan_auto_reply.infrastructure.llm import LegacyMentionChatBackend
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel
from shuiyuan_auto_reply.infrastructure.persistence import SQLiteStateStore
from shuiyuan_auto_reply.infrastructure.persistence import LocalSecretVault
from shuiyuan_auto_reply.infrastructure.prompts import FilePromptRepository
from shuiyuan_auto_reply.application.ports.prompt import PromptScope

DEEPSEEK_VISION_MODEL = "deepseek-v4-flash-vision-exp"


def _forum_profile_defaults(settings: AppSettings, persona: str) -> dict:
    prompt = FilePromptRepository().load(
        persona, set(), PromptScope.FORUM
    ).system_prompt
    return {
        "provider": "deepseek",
        "model": DEEPSEEK_VISION_MODEL,
        "api_format": DeepSeekApiFormat.CHAT_COMPLETIONS.value,
        "fallback_model": None,
        "system_prompt": prompt,
        "enabled_tools": None,
        "disabled_mcp_tools": [],
    }


async def _forum_provider_settings(
    settings: AppSettings, vault: LocalSecretVault, profile: dict
):
    provider = "deepseek"
    effective = replace(
        settings.providers,
        mention_provider=provider,
        deepseek_model=DEEPSEEK_VISION_MODEL,
        deepseek_api_format=DeepSeekApiFormat(
            profile.get("api_format", DeepSeekApiFormat.CHAT_COMPLETIONS.value)
        ),
    )
    secret = await vault.get(f"forum:{provider}")
    key_fields = {
        "openrouter": "openrouter_api_key",
        "deepseek": "deepseek_api_key",
        "tongyi": "dashscope_api_key",
        "mimo": "mimo_api_key",
    }
    if secret:
        effective = replace(effective, **{key_fields[provider]: secret})
    return effective


async def run_worker(persona: str = "wolf_lumine") -> None:
    load_dotenv()
    settings = AppSettings()
    state_store = SQLiteStateStore()
    await state_store.initialize()
    secret_vault = LocalSecretVault(state_store)
    model = await ShuiyuanModel.create(settings.forum.cookie_file)
    chat_model = None
    mention = None
    container = None
    try:
        profile = await state_store.get_profile(
            "forum", _forum_profile_defaults(settings, persona)
        )
        effective_settings = await _forum_provider_settings(
            settings, secret_vault, profile["active"]
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
            disabled_mcp_tools=set(
                profile["active"].get("disabled_mcp_tools", [])
            ),
            state_store=state_store,
            system_prompt_override=profile["active"].get("system_prompt"),
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
                    settings, secret_vault, latest["active"]
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
                    system_prompt_override=latest["active"].get("system_prompt"),
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

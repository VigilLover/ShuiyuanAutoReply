"""Factories are the only bootstrap code deciding concrete chat providers."""

from dataclasses import replace
from typing import Any

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    MentionDeepSeekModel,
)
from shuiyuan_auto_reply.features.mention.mention_mimo_model import MentionMimoModel
from shuiyuan_auto_reply.features.mention.mention_openrouter_model import (
    MentionOpenRouterModel,
)
from shuiyuan_auto_reply.features.mention.mention_tongyi_model import MentionTongyiModel
from shuiyuan_auto_reply.infrastructure.persistence.model_configs import (
    secret_name as model_config_secret_name,
)

from .settings import DeepSeekApiFormat, ProviderSettings


async def apply_profile_endpoint(
    settings: ProviderSettings,
    scope: str,
    profile: dict,
    *,
    store=None,
    vault=None,
) -> ProviderSettings:
    """Overlay the profile's endpoint, then whatever the active config pins.

    A stored configuration wins over the profile because switching one is meant
    to fully decide the endpoint, model, format and key for its scope.
    """
    overrides: dict[str, Any] = {
        "deepseek_model": (profile.get("model") or "").strip()
        or settings.deepseek_model,
        "mention_base_url": (profile.get("base_url") or "").strip() or None,
        "deepseek_api_format": DeepSeekApiFormat(
            profile.get("api_format") or settings.deepseek_api_format.value
        ),
    }
    if store is not None and scope in {"web", "forum"}:
        active = await store.active_model_config(scope)
        if active is not None:
            overrides["deepseek_model"] = active["model"] or overrides["deepseek_model"]
            overrides["mention_base_url"] = (
                active["base_url"] or overrides["mention_base_url"]
            )
            if active.get("api_format"):
                overrides["deepseek_api_format"] = DeepSeekApiFormat(
                    active["api_format"]
                )
            if vault is not None:
                api_key = (
                    await vault.get(model_config_secret_name(active["id"])) or ""
                ).strip()
                if api_key:
                    overrides["deepseek_api_key"] = api_key
    return replace(settings, **overrides)


class MentionProviderFactory:
    _providers = {
        "deepseek": MentionDeepSeekModel,
        "tongyi": MentionTongyiModel,
        "openrouter": MentionOpenRouterModel,
        "mimo": MentionMimoModel,
    }

    @classmethod
    def create(
        cls,
        forum_model,
        persona: str,
        settings: ProviderSettings,
        *,
        prompt_scope: PromptScope = PromptScope.FORUM,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
        prompt_profile: dict | None = None,
    ):
        from shuiyuan_auto_reply.infrastructure.prompts.profiles import render_profile

        if prompt_profile is not None:
            system_prompt_override = render_profile(prompt_profile, prompt_scope.value)
        settings.validate_forum()
        model = cls._providers[settings.mention_provider](
            forum_model,
            username=persona,
            provider_settings=settings,
            prompt_scope=prompt_scope,
            enabled_tools=enabled_tools,
            disabled_mcp_tools=disabled_mcp_tools,
            state_store=state_store,
            system_prompt_override=system_prompt_override,
        )
        from langchain_core.prompts import (
            ChatPromptTemplate,
            MessagesPlaceholder,
            SystemMessagePromptTemplate,
        )

        from shuiyuan_auto_reply.infrastructure.prompts.profiles import (
            fingerprint,
            profile_metadata,
        )

        if prompt_profile and prompt_profile.get("prompt_mode") == "managed":
            effective_prompt = render_profile(
                prompt_profile,
                prompt_scope.value,
                multimodal=bool(model._get_multimodal_prompt_rules()),
            )
            model.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(effective_prompt),
                    MessagesPlaceholder("chat_history"),
                    MessagesPlaceholder("messages"),
                ]
            )
        else:
            effective_prompt = model.prompt.messages[0].prompt.template
        model.runtime_profile_metadata = profile_metadata(
            prompt_profile or {"system_prompt": effective_prompt},
            prompt_scope.value,
        )
        model.runtime_profile_metadata["prompt_hash"] = fingerprint(effective_prompt)
        model.runtime_profile_metadata["profile_revision"] = (prompt_profile or {}).get(
            "profile_revision"
        )
        return model

    @staticmethod
    def create_api(
        forum_model,
        persona: str,
        settings: ProviderSettings,
        *,
        prompt_scope: PromptScope = PromptScope.WEB,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
        prompt_profile: dict | None = None,
    ):
        from shuiyuan_auto_reply.infrastructure.prompts.profiles import render_profile

        if prompt_profile is not None:
            system_prompt_override = render_profile(prompt_profile, prompt_scope.value)
        settings.validate_api()
        model = MentionOpenRouterModel(
            forum_model,
            username=persona,
            provider_settings=settings,
            prompt_scope=prompt_scope,
            enabled_tools=enabled_tools,
            disabled_mcp_tools=disabled_mcp_tools,
            state_store=state_store,
            system_prompt_override=system_prompt_override,
        )
        from langchain_core.prompts import (
            ChatPromptTemplate,
            MessagesPlaceholder,
            SystemMessagePromptTemplate,
        )

        from shuiyuan_auto_reply.infrastructure.prompts.profiles import (
            fingerprint,
            profile_metadata,
        )

        if prompt_profile and prompt_profile.get("prompt_mode") == "managed":
            effective_prompt = render_profile(
                prompt_profile,
                prompt_scope.value,
                multimodal=bool(model._get_multimodal_prompt_rules()),
            )
            model.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(effective_prompt),
                    MessagesPlaceholder("chat_history"),
                    MessagesPlaceholder("messages"),
                ]
            )
        else:
            effective_prompt = model.prompt.messages[0].prompt.template
        model.runtime_profile_metadata = profile_metadata(
            prompt_profile or {"system_prompt": effective_prompt},
            prompt_scope.value,
        )
        model.runtime_profile_metadata["prompt_hash"] = fingerprint(effective_prompt)
        model.runtime_profile_metadata["profile_revision"] = (prompt_profile or {}).get(
            "profile_revision"
        )
        return model

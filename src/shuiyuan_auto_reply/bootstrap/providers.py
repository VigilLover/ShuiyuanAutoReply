"""Factories are the only bootstrap code deciding concrete chat providers."""

from shuiyuan_auto_reply.application.ports.prompt import PromptScope
from shuiyuan_auto_reply.features.mention.mention_deepseek_model import (
    MentionDeepSeekModel,
)
from shuiyuan_auto_reply.features.mention.mention_mimo_model import MentionMimoModel
from shuiyuan_auto_reply.features.mention.mention_openrouter_model import (
    MentionOpenRouterModel,
)
from shuiyuan_auto_reply.features.mention.mention_tongyi_model import MentionTongyiModel

from .settings import ProviderSettings


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
        from shuiyuan_auto_reply.infrastructure.prompts.profiles import profile_metadata

        model.runtime_profile_metadata = profile_metadata(
            prompt_profile or {"system_prompt": system_prompt_override or ""},
            prompt_scope.value,
        )
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
        from shuiyuan_auto_reply.infrastructure.prompts.profiles import profile_metadata

        model.runtime_profile_metadata = profile_metadata(
            prompt_profile or {"system_prompt": system_prompt_override or ""},
            prompt_scope.value,
        )
        model.runtime_profile_metadata["profile_revision"] = (prompt_profile or {}).get(
            "profile_revision"
        )
        return model

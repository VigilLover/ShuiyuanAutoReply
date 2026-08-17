"""Factories are the only bootstrap code deciding concrete chat providers."""

from shuiyuan_auto_reply.features.mention.mention_deepseek_model import MentionDeepSeekModel
from shuiyuan_auto_reply.features.mention.mention_mimo_model import MentionMimoModel
from shuiyuan_auto_reply.features.mention.mention_openrouter_model import MentionOpenRouterModel
from shuiyuan_auto_reply.features.mention.mention_tongyi_model import MentionTongyiModel

from .settings import ProviderSettings
from shuiyuan_auto_reply.application.ports.prompt import PromptScope


class MentionProviderFactory:
    _providers = {
        "deepseek": MentionDeepSeekModel,
        "tongyi": MentionTongyiModel,
        "openrouter": MentionOpenRouterModel,
        "mimo": MentionMimoModel,
    }

    @classmethod
    def create(
        cls, forum_model, persona: str, settings: ProviderSettings, *,
        prompt_scope: PromptScope = PromptScope.FORUM,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
    ):
        settings.validate_forum()
        return cls._providers[settings.mention_provider](
            forum_model, username=persona, provider_settings=settings,
            prompt_scope=prompt_scope, enabled_tools=enabled_tools,
            disabled_mcp_tools=disabled_mcp_tools, state_store=state_store,
            system_prompt_override=system_prompt_override,
        )

    @staticmethod
    def create_api(
        forum_model, persona: str, settings: ProviderSettings, *,
        prompt_scope: PromptScope = PromptScope.WEB,
        enabled_tools: set[str] | None = None,
        disabled_mcp_tools: set[str] | None = None,
        state_store=None,
        system_prompt_override: str | None = None,
    ):
        settings.validate_api()
        return MentionOpenRouterModel(
            forum_model, username=persona, provider_settings=settings,
            prompt_scope=prompt_scope, enabled_tools=enabled_tools,
            disabled_mcp_tools=disabled_mcp_tools, state_store=state_store,
            system_prompt_override=system_prompt_override,
        )

"""Factories are the only bootstrap code deciding concrete chat providers."""

from shuiyuan_auto_reply.features.mention.mention_deepseek_model import MentionDeepSeekModel
from shuiyuan_auto_reply.features.mention.mention_mimo_model import MentionMimoModel
from shuiyuan_auto_reply.features.mention.mention_openrouter_model import MentionOpenRouterModel
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
    def create(cls, forum_model, persona: str, settings: ProviderSettings):
        settings.validate_forum()
        return cls._providers[settings.mention_provider](
            forum_model, username=persona, provider_settings=settings
        )

    @staticmethod
    def create_api(forum_model, persona: str, settings: ProviderSettings):
        settings.validate_api()
        return MentionOpenRouterModel(
            forum_model, username=persona, provider_settings=settings
        )

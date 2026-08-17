from typing import Dict, List

import langchain_core.utils.function_calling as function_calling
from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.openrouter.openrouter_model import (
    DEFAULT_OPENROUTER_MAX_RETRIES,
    OPENROUTER_BASE_URL,
    openrouter_async_http_client,
    openrouter_headers,
    openrouter_http_client,
)
from shuiyuan_auto_reply.bootstrap.settings import ProviderSettings
from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import MentionChatModel


def _register_openrouter_tool_types(tools: List[Dict[str, str]]) -> None:
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type and tool_type not in function_calling._WellKnownOpenAITools:
            function_calling._WellKnownOpenAITools = (
                *function_calling._WellKnownOpenAITools,
                tool_type,
            )


class MentionOpenRouterModel(MentionChatModel):
    """
    A model for managing OpenRouter-backed conversation data.
    """

    def __init__(
        self,
        model: ShuiyuanModel,
        username: str = "wolf_lumine",
        provider_settings: ProviderSettings | None = None,
    ):
        # Initialize the base class first to set up retriever and other components
        super().__init__(model, username=username)

        current = provider_settings or ProviderSettings()
        api_key = current.openrouter_api_key
        if not api_key:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

        self.openai_tools = [
            {
                "type": "openrouter:web_search",
            },
        ]
        _register_openrouter_tool_types(self.openai_tools)
        proxy = current.openrouter_proxy

        self.llm = ChatOpenAI(
            model=current.openrouter_mention_model,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.8,
            default_headers=openrouter_headers(),
            http_client=openrouter_http_client(proxy=proxy, trust_env=False),
            http_async_client=openrouter_async_http_client(proxy=proxy, trust_env=False),
            max_retries=DEFAULT_OPENROUTER_MAX_RETRIES,
        )

    def parse_model_output(self, raw_output: object) -> str:
        """
        Parse the raw output from the model to extract the final response text.

        :param raw_output: The raw output from the model.
        :return: The extracted response text.
        """
        if raw_output is None:
            return ""
        if isinstance(raw_output, str):
            return raw_output.strip()

        res = ""
        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict) and "text" in item:
                    res += item["text"]
                if hasattr(item, "text"):
                    res += item.text
                if isinstance(item, str):
                    res += item
        return res.strip()

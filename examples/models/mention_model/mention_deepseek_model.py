import os
from typing import Any

from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import FallbackLLM, MentionChatModel

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DEEPSEEK_FALLBACK_MODEL = "deepseek-v4-flash"
DEEPSEEK_DEFAULT_MAX_RETRIES = 3


def _mk_deepseek_llm(api_key: str, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        max_retries=DEEPSEEK_DEFAULT_MAX_RETRIES,
        # NOTE: thinking 模式与 LangChain tool-calling agent 不兼容。
        # DeepSeek 要求后续请求原样传回 reasoning_content，但 LangChain 会裁剪消息，
        # 导致 400: "The reasoning_content in the thinking mode must be passed back to the API."
        # 因此 Mention 模型（大量使用 tool calling）必须关闭 thinking。
        # Pet 模型无 tool calling，可安全开启 thinking。
        # reasoning_effort="max",
        # extra_body={"thinking": {"type": "enabled"}},
    )


class MentionDeepSeekModel(MentionChatModel):
    """
    A model for managing DeepSeek-backed conversation data.
    Falls back to deepseek-v4-flash if deepseek-v4-pro is unavailable.
    Both models use thinking mode with reasoning_effort=max.
    """

    def __init__(self, model: ShuiyuanModel, username: str = "wolf_lumine"):
        super().__init__(model, username=username)

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

        model_name = os.getenv("DEEPSEEK_MENTION_MODEL", DEEPSEEK_DEFAULT_MODEL)
        fallback_name = os.getenv("DEEPSEEK_MENTION_FALLBACK_MODEL", DEEPSEEK_FALLBACK_MODEL)

        self.llm = FallbackLLM(
            _mk_deepseek_llm(api_key, model_name),
            _mk_deepseek_llm(api_key, fallback_name),
        )

    def parse_model_output(self, raw_output: Any) -> str:
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

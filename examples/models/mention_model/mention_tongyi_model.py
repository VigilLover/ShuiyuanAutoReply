import os

from langchain_openai import ChatOpenAI

from shuiyuan_auto_reply.shuiyuan.shuiyuan_model import ShuiyuanModel

from .mention_chat_model import FallbackLLM, MentionChatModel

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_DEFAULT_MODEL = "qwen3.5-plus-2026-02-15"
DASHSCOPE_FALLBACK_MODEL = "qwen3.5-plus"
# DashScope 特有的 thinking/增量输出参数，通过 extra_body 传入
DASHSCOPE_EXTRA_BODY = {
    "enable_thinking": True,
    "incremental_output": True,
}


def _mk_tongyi_llm(api_key: str, model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=DASHSCOPE_BASE_URL,
        temperature=1.5,
        extra_body=DASHSCOPE_EXTRA_BODY,
    )


class MentionTongyiModel(MentionChatModel):
    """
    A model for managing Tongyi Qianwen data via the OpenAI-compatible DashScope endpoint.
    Falls back to a secondary model internally if the primary is unavailable.
    """

    def __init__(self, model: ShuiyuanModel, username: str = "wolf_lumine"):
        super().__init__(model, username=username)

        api_key = os.getenv("DASHSCOPE_API_KEY")
        model_name = os.getenv("DASHSCOPE_MENTION_MODEL", DASHSCOPE_DEFAULT_MODEL)
        fallback_name = os.getenv("DASHSCOPE_MENTION_FALLBACK_MODEL", DASHSCOPE_FALLBACK_MODEL)

        self.llm = FallbackLLM(
            _mk_tongyi_llm(api_key, model_name),
            _mk_tongyi_llm(api_key, fallback_name),
        )

    def parse_model_output(self, raw_output: str) -> str:
        """
        Parse the raw output from the model to extract the final response text.

        :param raw_output: The raw output from the model, which is expected to be a string.
        :return: The extracted response text.
        """
        return raw_output.strip()

from shuiyuan_auto_reply.domain.capabilities import ProviderCapabilities


class LangChainLLMProvider:
    def __init__(self, model, capabilities: ProviderCapabilities) -> None:
        self.model = model
        self._capabilities = capabilities

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def invoke(self, messages, tools):
        bound = self.model.bind_tools(tools) if tools else self.model
        return await bound.ainvoke(messages)

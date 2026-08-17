"""Named Chat pipeline components around the behavior-frozen LangGraph nodes.

The first refactor phase delegates to the characterized implementation. Each
component is independently replaceable and testable without changing graph
ordering, conditional edges, retries, or retrieval parameters.
"""


class _Node:
    def __init__(self, owner) -> None:
        self.owner = owner


class StyleContextLoader(_Node):
    async def __call__(self, state):
        return await self.owner._retrieve_style_context(state)


class ChannelContextLoader(_Node):
    async def __call__(self, state):
        return await self.owner._load_topic_context(state)


class LongTermMemoryLoader(_Node):
    async def __call__(self, state):
        return await self.owner._load_long_term_memory(state)


class MultimodalInputLoader(_Node):
    async def current(self, state):
        return await self.owner._load_current_images(state)

    async def replied(self, state):
        return await self.owner._load_replied_post_images(state)

    async def tool_outputs(self, state):
        return await self.owner._collect_tool_output_images(state)


class MessagePreparer(_Node):
    async def __call__(self, state):
        return await self.owner._prepare_messages(state)


class AgentRuntime(_Node):
    async def __call__(self, state):
        return await self.owner._call_model(state)


class ToolCallValidator(_Node):
    async def __call__(self, state):
        return await self.owner._validate_tool_calls(state)


class ResponseParser(_Node):
    async def __call__(self, state):
        return await self.owner._finalize_response(state)


class HistoryWriter(_Node):
    async def __call__(self, state):
        return await self.owner._save_history(state)


class ChatOrchestrator:
    def __init__(self, owner) -> None:
        self.style = StyleContextLoader(owner)
        self.channel = ChannelContextLoader(owner)
        self.memory = LongTermMemoryLoader(owner)
        self.multimodal = MultimodalInputLoader(owner)
        self.messages = MessagePreparer(owner)
        self.runtime = AgentRuntime(owner)
        self.tool_validator = ToolCallValidator(owner)
        self.response = ResponseParser(owner)
        self.history = HistoryWriter(owner)

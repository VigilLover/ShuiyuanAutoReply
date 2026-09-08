from shuiyuan_auto_reply.application.ports.memory import MemoryCommand, MemoryScope


class PostgresLongTermMemoryAdapter:
    """Port adapter over the current LangMem-backed mention memory model."""

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    async def search(self, scope: MemoryScope, query: str, limit: int) -> str:
        return await self.delegate.search_mention_memory(
            target_user_id=scope.namespace, query=query, limit=limit
        )

    async def manage(self, command: MemoryCommand) -> str:
        return await self.delegate.manage_mention_memory(**command.payload)

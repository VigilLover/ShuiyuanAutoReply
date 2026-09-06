"""Adapter that preserves the existing Neo4j search call exactly."""

from collections.abc import Awaitable, Callable

from shuiyuan_auto_reply.application.ports.retrieval import StyleExample
async def create_global_async_neo4j_manager():
    from shuiyuan_auto_reply.database.neo4j_mgr import create_global_async_neo4j_manager as create
    return await create()


async def close_neo4j():
    import sys
    module = sys.modules.get("shuiyuan_auto_reply.database.neo4j_mgr")
    if module is not None:
        await module.close_global_async_neo4j_manager()



class Neo4jStyleRetriever:
    def __init__(
        self,
        manager_factory: Callable[[], Awaitable[object | None]] = create_global_async_neo4j_manager,
    ) -> None:
        self._manager_factory = manager_factory

    async def search(
        self, persona_id: str, query: str, limit: int
    ) -> list[StyleExample]:
        manager = await self._manager_factory()
        if manager is None:
            return []
        results = await manager.search_similar(query, top_k=limit, userid=persona_id)
        return [
            StyleExample(item.text, getattr(item, "score", None)) for item in results
        ]

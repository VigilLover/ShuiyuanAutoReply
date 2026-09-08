"""Run using the previous release's installed package against the migrated test DB."""

import asyncio

from shuiyuan_auto_reply.bootstrap.deployment import load_deployment
from shuiyuan_auto_reply.infrastructure.operations.migration import memory_store
from shuiyuan_auto_reply.infrastructure.persistence.state import SQLiteStateStore
from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
    check_vector_space,
    engine_for,
)


async def main():
    load_deployment("/validation/config.toml", "remote")
    engine = engine_for()
    try:
        async with engine.connect() as connection:
            await check_vector_space(connection)
            from sqlalchemy import text

            assert (
                await connection.execute(text("SELECT count(*) FROM style_sentences"))
            ).scalar_one() == 2
        async with memory_store(indexed=False) as store:
            assert await store.aget(("mention_memories", "test-user"), "test-key")
        await SQLiteStateStore().initialize()
    finally:
        await engine.dispose()


asyncio.run(main())

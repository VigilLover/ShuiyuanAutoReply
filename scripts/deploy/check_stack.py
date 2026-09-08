#!/usr/bin/env python3
"""Synthetic integration/soak checks: real PostgreSQL/MCP, fake model and forum.

Never calls paid model services or publishes forum replies. Run in compose.test.yaml.
"""

import asyncio
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        vector = [float(digest[i % 32]) / 255 for i in range(1024)]
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]


async def main():
    import base64

    from langchain_mcp_adapters.client import MultiServerMCPClient

    from shuiyuan_auto_reply.application.scheduling import ReplyScheduler
    from shuiyuan_auto_reply.bootstrap.deployment import load_deployment
    from shuiyuan_auto_reply.features.mention.deepseek_vision import save_uploaded_image
    from shuiyuan_auto_reply.infrastructure import embedding
    from shuiyuan_auto_reply.infrastructure.operations.migration import (
        export_memory,
        import_data,
        memory_store,
        migrate_database,
    )
    from shuiyuan_auto_reply.infrastructure.persistence.state import SQLiteStateStore
    from shuiyuan_auto_reply.infrastructure.persistence.work_queue import ForumQueue
    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
        PostgresStyleRetriever,
    )

    config_path = Path("/tmp/validation.toml")
    config_path.write_text("""[profiles.remote.embedding]
backend="openai"
model="synthetic-qwen3.7"
dims=1024
base_url="http://fake:8080/v1"
api_key="synthetic"
[profiles.remote.database]
url={env="VALIDATION_DB_URL"}
migration_url={env="VALIDATION_ADMIN_URL"}
auto_migrate=false
[profiles.remote.paths]
state_dir="/tmp/validation-state"
""")
    config = load_deployment(str(config_path), "remote")
    embedding._instance = DeterministicEmbeddings()
    embedding._fingerprint = config.fingerprint
    await migrate_database()
    from sqlalchemy import text

    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import engine_for

    engine = engine_for()
    try:
        async with engine.begin() as connection:
            try:
                await connection.execute(
                    text("CREATE TABLE forbidden_runtime_ddl (id int)")
                )
            except Exception as error:
                assert "permission denied" in str(error).lower()
            else:
                raise AssertionError("Runtime role can execute DDL")
    finally:
        await engine.dispose()
    source = Path("/tmp/corpus.jsonl")
    source.write_text(
        json.dumps({"persona_id": "wolf", "text": "hello wolf"})
        + "\n"
        + json.dumps({"persona_id": "other", "text": "hello other"})
        + "\n"
    )
    assert await import_data("corpus", source) == 2
    assert await import_data("corpus", source) == 0
    memory = Path("/tmp/memory.jsonl")
    memory.write_text(
        json.dumps(
            {
                "namespace": ["mention_memories", "test-user"],
                "key": "test-key",
                "value": {"content": "likes science"},
            }
        )
        + "\n"
    )
    assert await import_data("memory", memory) == 1
    assert await import_data("memory", memory) == 0
    out = Path("/tmp/export.jsonl")
    assert await export_memory(out) == 1
    retriever = PostgresStyleRetriever()
    assert len(await retriever.search("wolf", "hello", 8)) == 1
    assert not await retriever.search("missing", "hello", 8)
    store = SQLiteStateStore(Path("/tmp/validation-state/state.sqlite3"))
    await store.initialize()
    queue = ForumQueue(store.path, "synthetic")
    await queue.initialize()
    db = await queue.connect()
    await db.execute(
        "INSERT INTO forum_jobs VALUES ('synthetic',1,'{}','sending',NULL,0)"
    )
    await db.commit()
    await db.close()
    await queue.initialize()
    assert await queue.state(1) == "needs_review"
    client = MultiServerMCPClient(
        {"tools": {"url": "http://mcp:58000/sse", "transport": "sse"}}
    )
    tools = await client.get_tools()
    clock = next(tool for tool in tools if tool.name == "get_system_time")
    assert await clock.ainvoke({})
    scheduler = ReplyScheduler(3, 100, 900)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    from shuiyuan_auto_reply.domain import Channel, ConversationRef

    conversations = [
        await store.ensure_conversation(
            ConversationRef(Channel.WEB, str(i), "bot", "wolf")
        )
        for i in range(3)
    ]

    async def work(index):
        async with scheduler.admission(str(index)):
            assert await retriever.search("wolf", "hello", 8)
            await save_uploaded_image(
                store,
                conversation_id=conversations[index].id,
                data=png,
                filename="test.png",
            )
            await asyncio.sleep(0.05)

    duration = int(os.environ.get("VALIDATION_SECONDS", "60"))
    start = time.monotonic()
    iterations = 0
    try:
        while time.monotonic() - start < duration:
            await asyncio.gather(*(work(i) for i in range(3)))
            iterations += 1
            if iterations % 30 == 0:
                print(
                    json.dumps(
                        {
                            "elapsed": round(time.monotonic() - start),
                            "iterations": iterations,
                            "max_rss_kib": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss,
                        }
                    ),
                    flush=True,
                )
                assert await clock.ainvoke({})
            await asyncio.sleep(1)
    finally:
        await retriever.aclose()
    print(
        json.dumps(
            {
                "status": "passed",
                "seconds": duration,
                "iterations": iterations,
                "synthetic": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

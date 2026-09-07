"""Streaming, restartable data transfer. Sources are never modified."""

import csv
import hashlib
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
from shuiyuan_auto_reply.constants import settings


from shuiyuan_auto_reply.application.operations.migration import write_json, digest


@asynccontextmanager
async def memory_store(*, indexed=True, migration=False):
    from langgraph.store.postgres.aio import AsyncPostgresStore

    from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings
    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import database_url

    config = get_deployment()
    kwargs = {}
    if indexed:
        kwargs["index"] = dict(
            dims=config.section("embedding")["dims"],
            embed=get_embeddings(),
            fields=["content"],
        )
    url = database_url("memory_url", migration=migration).replace(
        "postgresql+psycopg://", "postgresql://"
    )
    async with AsyncPostgresStore.from_conn_string(url, **kwargs) as store:
        yield store


async def migrate_database():
    from shuiyuan_auto_reply.database.postgres_memory_mgr import MemoryPostgresBase
    from shuiyuan_auto_reply.database.postgres_record_mgr import RecordPostgresBase
    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
        engine_for,
        initialize_vector_schema,
    )

    engine = engine_for(migration=True)
    try:
        async with engine.begin() as connection:
            await initialize_vector_schema(connection)
            await connection.run_sync(MemoryPostgresBase.metadata.create_all)
            await connection.run_sync(RecordPostgresBase.metadata.create_all)
        async with memory_store(migration=True) as store:
            await store.setup()
    finally:
        await engine.dispose()


async def export_corpus(output, *, csv_path=None, persona=None):
    count = 0
    with Path(output).open("x", encoding="utf-8") as target:
        if csv_path:
            if not persona:
                raise ValueError("--persona is required for CSV input")
            with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    raw = row.get("post_raw") or row.get("text") or ""
                    if not raw.strip() or settings.contains_auto_reply_tag(raw):
                        continue
                    content = re.sub(
                        r"<div data-signature>.*?</div>", "", raw, flags=re.DOTALL
                    ).strip()
                    target.write(
                        json.dumps(
                            dict(persona_id=persona, text=content), ensure_ascii=False
                        )
                        + "\n"
                    )
                    count += 1
        else:
            import ast

            from neo4j import AsyncGraphDatabase

            config = get_deployment().section("neo4j")
            auth = ast.literal_eval(config["auth"]) if config["auth"] else None
            async with AsyncGraphDatabase.driver(config["url"], auth=auth) as driver:
                async with driver.session(default_access_mode="READ") as session:
                    result = await session.run(
                        "MATCH (n:Sentence) WHERE ($persona IS NULL OR n.userid=$persona) RETURN n.userid AS persona_id,n.text AS text",
                        persona=persona,
                    )
                    async for row in result:
                        if not row["persona_id"]:
                            raise ValueError(
                                "Legacy corpus has no persona_id; repair source ownership first"
                            )
                        target.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                        count += 1
    return count


async def export_memory(output):
    count = 0
    async with memory_store(indexed=False) as store:
        with Path(output).open("x", encoding="utf-8") as target:
            offset = 0
            while True:
                items = await store.asearch((), limit=100, offset=offset)
                if not items:
                    break
                for item in items:
                    target.write(
                        json.dumps(
                            dict(
                                namespace=list(item.namespace),
                                key=item.key,
                                value=item.value,
                                source_created_at=str(item.created_at),
                                source_updated_at=str(item.updated_at),
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
                offset += len(items)
    return count


async def import_data(kind, source, *, dry_run=False):
    from shuiyuan_auto_reply.infrastructure.retrieval.postgres import (
        PostgresStyleRetriever,
        check_vector_space,
        engine_for,
    )

    source = Path(source)
    checkpoint = source.with_suffix(source.suffix + f".{kind}.checkpoint.json")
    failures = source.with_suffix(source.suffix + f".{kind}.failures.json")
    identity = dict(
        source=digest(source), fingerprint=get_deployment().fingerprint, kind=kind
    )
    progress = dict(identity, completed=[])
    if checkpoint.exists():
        progress = json.loads(checkpoint.read_text())
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError("Checkpoint belongs to different source or vector space")
    completed = set(progress["completed"])
    errors = []
    count = 0
    engine = engine_for()
    retriever = None
    try:
        if not dry_run:
            async with engine.connect() as connection:
                await check_vector_space(connection)
        async with optional_memory_store(kind == "memory" and not dry_run) as store:
            if kind == "corpus" and not dry_run:
                retriever = PostgresStyleRetriever(engine=engine)
            with source.open(encoding="utf-8") as lines:
                for number, line in enumerate(lines, 1):
                    if number in completed:
                        continue
                    try:
                        record = json.loads(line)
                        if kind == "corpus":
                            if (
                                not record.get("persona_id")
                                or not record.get("text", "").strip()
                            ):
                                raise ValueError(
                                    "persona_id and nonempty text required"
                                )
                            content = record["text"]
                            if settings.contains_auto_reply_tag(content):
                                content = ""
                            content = re.sub(
                                r"<div data-signature>.*?</div>",
                                "",
                                content,
                                flags=re.DOTALL,
                            ).strip()
                            if not dry_run and content:
                                await retriever.store(record["persona_id"], content)
                        else:
                            if (
                                not isinstance(record.get("namespace"), list)
                                or not record["namespace"]
                                or not all(
                                    isinstance(x, str) for x in record["namespace"]
                                )
                                or not record.get("key")
                                or not isinstance(record.get("value"), dict)
                            ):
                                raise ValueError("Invalid memory namespace/key/value")
                            if not dry_run:
                                await store.aput(
                                    tuple(record["namespace"]),
                                    record["key"],
                                    record["value"],
                                )
                                from shuiyuan_auto_reply.database.postgres_memory_mgr import (
                                    AsyncPostgresMemoryDatabaseManager,
                                )

                                manager = AsyncPostgresMemoryDatabaseManager()
                                try:
                                    await manager.touch_mention_memory_key(
                                        record["namespace"][-1]
                                    )
                                finally:
                                    await manager.close()
                        count += 1
                        if not dry_run:
                            completed.add(number)
                            progress["completed"] = sorted(completed)
                            write_json(checkpoint, progress)
                    except Exception as exc:
                        # Never persist provider errors that may contain credentials or text.
                        errors.append(dict(line=number, error=type(exc).__name__))
            if not dry_run:
                write_json(failures, errors)
    finally:
        await engine.dispose()
    if errors:
        raise ValueError(
            f"{len(errors)} records failed; inspect failure report (dry-run does not write reports): {errors[:10]}"
        )
    return count


@asynccontextmanager
async def optional_memory_store(enabled):
    if enabled:
        async with memory_store() as store:
            yield store
    else:
        yield None

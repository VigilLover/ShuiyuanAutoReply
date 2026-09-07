"""Small-corpus exact pgvector retrieval and vector-space compatibility checks."""

import hashlib
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from shuiyuan_auto_reply.application.ports.retrieval import StyleExample
from shuiyuan_auto_reply.bootstrap.deployment import get_deployment
from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings


def database_url(kind="url", *, migration=False):
    config = get_deployment().section("database")
    url = (
        (config["migration_url"] if migration else "")
        or config.get(kind)
        or config["url"]
    )
    if not url:
        raise ValueError("PostgreSQL URL is required")
    return url.replace("postgresql://", "postgresql+psycopg://", 1).replace(
        "postgres://", "postgresql+psycopg://", 1
    )


def engine_for(kind="url", *, migration=False):
    return create_async_engine(
        database_url(kind, migration=migration),
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )


async def check_vector_space(connection):
    result = await connection.execute(
        text("SELECT fingerprint FROM deployment_vector_space WHERE singleton=1")
    )
    if result.scalar_one_or_none() != get_deployment().fingerprint:
        raise RuntimeError(
            "Vector space mismatch: re-embed into a separate database before startup"
        )


async def initialize_vector_schema(connection):
    config = get_deployment()
    dims = int(config.section("embedding")["dims"])
    await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS deployment_vector_space (singleton integer PRIMARY KEY CHECK(singleton=1), fingerprint text NOT NULL)"
        )
    )
    await connection.execute(
        text(
            "INSERT INTO deployment_vector_space VALUES (1,:fp) ON CONFLICT DO NOTHING"
        ),
        {"fp": config.fingerprint},
    )
    await check_vector_space(connection)
    await connection.execute(
        text(
            f"""CREATE TABLE IF NOT EXISTS style_sentences (
        persona_id text NOT NULL, text_hash text NOT NULL, text text NOT NULL,
        embedding vector({dims}) NOT NULL, embedding_version text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(persona_id,text_hash))"""
        )
    )
    await connection.execute(
        text("CREATE INDEX IF NOT EXISTS style_persona ON style_sentences(persona_id)")
    )


class PostgresStyleRetriever:
    def __init__(self, embedding=None, engine=None):
        self.embedding = embedding or get_embeddings()
        self.engine = engine or engine_for()

    async def search(self, persona_id, query, limit):
        if limit <= 0:
            return []
        vector = await self.embedding.aembed_query(query)
        async with self.engine.connect() as connection:
            await check_vector_space(connection)
            rows = await connection.execute(
                text("""SELECT text,
                1-(embedding <=> CAST(:vector AS vector)) AS score FROM style_sentences
                WHERE persona_id=:persona ORDER BY embedding <=> CAST(:vector AS vector)
                LIMIT :limit"""),
                dict(vector=json.dumps(vector), persona=persona_id, limit=limit),
            )
            return [StyleExample(row.text, row.score) for row in rows]

    async def store(self, persona_id, content):
        vector = await self.embedding.aembed_query(content)
        async with self.engine.begin() as connection:
            await check_vector_space(connection)
            await connection.execute(
                text("""INSERT INTO style_sentences
                (persona_id,text_hash,text,embedding,embedding_version)
                VALUES (:persona,:hash,:content,CAST(:vector AS vector),:version)
                ON CONFLICT (persona_id,text_hash) DO NOTHING"""),
                dict(
                    persona=persona_id,
                    hash=hashlib.sha256(content.encode()).hexdigest(),
                    content=content,
                    vector=json.dumps(vector),
                    version=get_deployment().fingerprint,
                ),
            )

    async def aclose(self):
        await self.engine.dispose()

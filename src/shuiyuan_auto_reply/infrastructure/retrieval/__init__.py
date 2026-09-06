"""Configuration-driven style retrieval composition."""
from shuiyuan_auto_reply.bootstrap.deployment import get_deployment


class DisabledStyleRetriever:
    async def search(self, persona_id, query, limit):
        return []


def create_style_retriever():
    backend = get_deployment().section('retrieval')['backend']
    if backend == 'pgvector':
        from .postgres import PostgresStyleRetriever
        return PostgresStyleRetriever()
    if backend == 'neo4j':
        from .neo4j import Neo4jStyleRetriever
        return Neo4jStyleRetriever()
    return DisabledStyleRetriever()

from .neo4j import Neo4jStyleRetriever  # compatibility export; lazy SDK import

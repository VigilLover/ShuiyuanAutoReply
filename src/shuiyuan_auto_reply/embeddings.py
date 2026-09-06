"""Compatibility facade; importing this module never imports local ML dependencies."""
from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings


def get_global_text_embeddings():
    return get_embeddings()


def get_global_sentence_transformer():
    from shuiyuan_auto_reply.infrastructure.embedding.local import SharedTextEmbeddings
    return SharedTextEmbeddings.get_sentence_transformer()


def __getattr__(name):
    if name == 'SharedTextEmbeddings':
        from shuiyuan_auto_reply.infrastructure.embedding.local import SharedTextEmbeddings
        return SharedTextEmbeddings
    raise AttributeError(name)

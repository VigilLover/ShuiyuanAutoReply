"""Embedding adapters selected at the composition boundary."""
from shuiyuan_auto_reply.bootstrap.deployment import get_deployment

_instance = None
_fingerprint = None


def get_embeddings():
    global _instance, _fingerprint
    config = get_deployment()
    if _instance is not None and _fingerprint != config.fingerprint:
        raise RuntimeError('Embedding configuration changed; restart the process')
    if _instance is None:
        if config.section('embedding')['backend'] == 'local':
            from .local import SharedTextEmbeddings
            _instance = SharedTextEmbeddings()
        else:
            from .openai import OpenAIEmbeddings
            _instance = OpenAIEmbeddings(config.section('embedding'))
        _fingerprint = config.fingerprint
    return _instance


async def close_embeddings():
    global _instance, _fingerprint
    instance, _instance = _instance, None
    _fingerprint = None
    if instance is not None and hasattr(instance, 'aclose'):
        await instance.aclose()

"""Check extras without loading model weights or contacting providers."""

import importlib.util
import sys


def main():
    local = sys.argv[1] == "local"
    for name in ("torch", "sentence_transformers", "neo4j", "neomodel"):
        assert (importlib.util.find_spec(name) is not None) == local, name
    from shuiyuan_auto_reply.infrastructure.embedding import get_embeddings
    from shuiyuan_auto_reply.infrastructure.embedding.openai import OpenAIEmbeddings

    if local:
        from shuiyuan_auto_reply.infrastructure.embedding.local import (
            SharedTextEmbeddings,
        )

    print("Backend dependency contract passed")


if __name__ == "__main__":
    main()

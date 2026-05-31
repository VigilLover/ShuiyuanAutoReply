import logging
import threading
from _thread import LockType
from typing import ClassVar, List, Optional, cast

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from shuiyuan_auto_reply.constants import settings


class SharedTextEmbeddings(Embeddings):
    """Singleton LangChain embedding wrapper for the configured text model."""

    _instance: ClassVar[Optional["SharedTextEmbeddings"]] = None
    _model: ClassVar[Optional[SentenceTransformer]] = None
    _model_name: ClassVar[Optional[str]] = None
    _lock: ClassVar[LockType] = threading.Lock()

    def __new__(cls) -> "SharedTextEmbeddings":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cast("SharedTextEmbeddings", super().__new__(cls))
        return cls._instance

    @classmethod
    def _load_sentence_transformer(cls, model_name: str) -> SentenceTransformer:
        cache_folder = settings.embedding_cache_folder
        try:
            return SentenceTransformer(
                model_name,
                cache_folder=cache_folder,
                local_files_only=True,
            )
        except Exception as exc:
            logging.info(
                "Embedding model %r was not available from local cache; "
                "falling back to Hugging Face download. Error: %s",
                model_name,
                exc,
            )
            return SentenceTransformer(model_name, cache_folder=cache_folder)

    @classmethod
    def get_sentence_transformer(cls) -> SentenceTransformer:
        model_name = settings.embedding_model_name
        with cls._lock:
            model = cls._model
            if model is None:
                cls._model_name = model_name
                model = cls._load_sentence_transformer(model_name)
                cls._model = model
            elif cls._model_name != model_name:
                raise RuntimeError(
                    "Embedding model is already initialized as "
                    f"{cls._model_name!r}; cannot switch to {model_name!r} in-process."
                )
            return model

    @property
    def model(self) -> SentenceTransformer:
        return self.get_sentence_transformer()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


def get_global_text_embeddings() -> SharedTextEmbeddings:
    return SharedTextEmbeddings()


def get_global_sentence_transformer() -> SentenceTransformer:
    return SharedTextEmbeddings.get_sentence_transformer()

"""Turn text into vectors with one of the supported multilingual models."""

from functools import lru_cache
from typing import Sequence

import numpy as np

from parakh.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS


@lru_cache(maxsize=len(EMBEDDING_MODELS))
def _load_model(model: str):
    """Load a model once and reuse it (heavy imports stay lazy)."""
    if model == "BAAI/bge-m3":
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(model, use_fp16=False)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model)


def embed(text: str | Sequence[str], model: str = DEFAULT_EMBEDDING_MODEL) -> np.ndarray:
    """A string gives a 1-D vector, a list of strings gives one row per string."""
    if model not in EMBEDDING_MODELS:
        raise ValueError(f"Unsupported embedding model: {model}")
    single = isinstance(text, str)
    texts = [text] if single else list(text)
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    encoder = _load_model(model)
    if model == "BAAI/bge-m3":
        result = encoder.encode(texts, batch_size=12, max_length=512)["dense_vecs"]
    else:
        result = encoder.encode(texts)
    vectors = np.asarray(result, dtype=np.float32)
    return vectors[0] if single else vectors


def to_pgvector(vector: Sequence[float]) -> str:
    """Format a vector as pgvector input text: [0.1,0.2,...]."""
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def embed_unique(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, str]:
    """Embed each distinct text once (answers like 'Yes'/'हाँ' repeat a lot).

    Returns {text: pgvector literal}.
    """
    unique = list(dict.fromkeys(texts))
    if not unique:
        return {}
    vectors = embed(unique, model)
    return {text: to_pgvector(vector) for text, vector in zip(unique, vectors)}

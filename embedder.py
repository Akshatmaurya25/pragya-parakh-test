
"""Create text embeddings with one of the supported multilingual models."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from FlagEmbedding import BGEM3FlagModel

MODEL_OPTIONS = {
    "1": ("bge-m3", "BAAI/bge-m3"),
    "2": ("krutrim-ai-labs/vyakyarth", "krutrim-ai-labs/vyakyarth"),
    "3": ("google/embeddinggemma-300m", "google/embeddinggemma-300m"),
}
MODEL_STORAGE_NAMES = {
    "BAAI/bge-m3": "bge-m3",
    "krutrim-ai-labs/vyakyarth": "krutrim-ai-labs/vyakyarth",
    "google/embeddinggemma-300m": "google/embeddinggemma-300m",
}
DEFAULT_MODEL_NAME = "krutrim-ai-labs/vyakyarth"
MODEL_NAME = DEFAULT_MODEL_NAME


@lru_cache(maxsize=3)
def _load_model(model_name: str) -> SentenceTransformer | BGEM3FlagModel:
    """Load the requested embedding model once and reuse it."""
    if model_name == "BAAI/bge-m3":
        from FlagEmbedding import BGEM3FlagModel

        return BGEM3FlagModel(model_name, use_fp16=False)
    return SentenceTransformer(model_name)


def embedder(
    text: str | Sequence[str],
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """
    Embed a string or sequence of strings.

    A string returns a 1-D array. A sequence returns a 2-D array with one
    vector per input string. Vector dimensions depend on the selected model.
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    if not all(isinstance(item, str) for item in texts):
        raise TypeError("text must be a string or a sequence of strings")

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    supported_model_names = {
        name for _, name in MODEL_OPTIONS.values()
    }
    if model_name not in supported_model_names:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    model = _load_model(model_name)
    if model_name == "BAAI/bge-m3":
        result = model.encode(
            texts,
            batch_size=12,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
    else:
        result = model.encode(texts)

    vectors = np.asarray(result, dtype=np.float32)
    return vectors[0] if single else vectors

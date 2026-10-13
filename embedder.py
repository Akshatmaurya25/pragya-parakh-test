
"""Create multilingual sentence embeddings for English and Hindi text."""

from functools import lru_cache
from typing import Sequence

import numpy as np
from FlagEmbedding import BGEM3FlagModel


@lru_cache(maxsize=1)
def _load_model() -> BGEM3FlagModel:
    """Load the model once and reuse it."""
    return BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=False,
    )


def embedder(text: str | Sequence[str]) -> np.ndarray:
    """
    Embed a string or sequence of strings.

    A string returns a 1-D array of shape (1024,).
    A sequence returns a 2-D array of shape (n, 1024).
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    if not all(isinstance(item, str) for item in texts):
        raise TypeError("text must be a string or a sequence of strings")

    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    result = _load_model().encode(
        texts,
        batch_size=12,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    vectors = np.asarray(result["dense_vecs"], dtype=np.float32)

    return vectors[0] if single else vectors

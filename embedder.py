
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "google/embeddinggemma-300m"

@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the model once and reuse it."""
    model = SentenceTransformer(MODEL_NAME)
    return SentenceTransformer("google/embeddinggemma-300m")


def embedder(text: str | Sequence[str]) -> np.ndarray:
    """
    A single string returns shape (dimensions,).
    A sequence returns shape (n, dimensions).
    """
    single = isinstance(text, str)
    texts = [text] if single else list(text)

    if not all(isinstance(item, str) for item in texts):
        raise TypeError(
            "text must be a string or a sequence of strings"
        )

    if not texts:
        dimension = _load_model().get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)

    vectors = _load_model().encode(
        texts,
        batch_size=12,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    return vectors[0] if single else vectors

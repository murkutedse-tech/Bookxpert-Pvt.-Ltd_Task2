"""Step 3 - Embeddings with a local sentence-transformers model.

All chunks are embedded in batched calls (`model.encode(list_of_texts,
batch_size=...)`), never one chunk at a time. Vectors are L2-normalised so
that an inner product equals cosine similarity.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

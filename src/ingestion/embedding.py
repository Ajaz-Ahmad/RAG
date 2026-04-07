import logging

import numpy as np
from openai import OpenAI

from config import Settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 256


class Embedder:
    """Wraps an OpenAI-compatible embeddings endpoint (vLLM, OpenAI, etc.).

    The model name is auto-detected from the server if not set in config,
    which makes this work out-of-the-box against a local vLLM instance.
    """

    def __init__(self, settings: Settings) -> None:
        self.client = OpenAI(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
        if settings.embedding_model:
            self.model = settings.embedding_model
        else:
            available = self.client.models.list().data
            if not available:
                raise RuntimeError("No models found on the embedding server.")
            self.model = available[0].id
            logger.info("Auto-detected embedding model: %s", self.model)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts in batches. Returns shape (N, dim)."""
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self.client.embeddings.create(model=self.model, input=batch)
            # API returns embeddings ordered by index
            ordered = sorted(response.data, key=lambda e: e.index)
            all_embeddings.extend(e.embedding for e in ordered)
            logger.debug("Embedded batch %d/%d", i // _BATCH_SIZE + 1, -(-len(texts) // _BATCH_SIZE))

        return np.array(all_embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single string. Returns shape (dim,)."""
        return self.embed([text])[0]

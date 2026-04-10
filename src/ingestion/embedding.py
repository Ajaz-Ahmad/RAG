import logging

import numpy as np
from openai import OpenAI

from config import Settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 256


class Embedder:
    """Wraps either a local sentence-transformers model or an OpenAI-compatible API.

    Local mode (default):  uses sentence-transformers — free, no API key required.
    API mode:              uses any OpenAI-compatible endpoint (OpenAI, vLLM, etc.).

    Controlled by settings.use_local_embeddings.
    """

    def __init__(self, settings: Settings) -> None:
        if settings.use_local_embeddings:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local embedding model '%s'…", settings.local_embedding_model)
            self._local = SentenceTransformer(settings.local_embedding_model)
            self._mode = "local"
            logger.info("Local embedding model ready.")
        else:
            self._client = OpenAI(
                base_url=settings.embedding_base_url,
                api_key=settings.embedding_api_key,
            )
            if settings.embedding_model:
                self.model = settings.embedding_model
            else:
                available = self._client.models.list().data
                if not available:
                    raise RuntimeError("No models found on the embedding server.")
                self.model = available[0].id
                logger.info("Auto-detected embedding model: %s", self.model)
            self._mode = "api"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns shape (N, dim)."""
        if not texts:
            raise ValueError("Cannot embed an empty list.")

        if self._mode == "local":
            vecs = self._local.encode(texts, batch_size=_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
            return np.array(vecs, dtype=np.float32)

        # API mode — batch to stay under token limits
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = self._client.embeddings.create(model=self.model, input=batch)
            ordered = sorted(response.data, key=lambda e: e.index)
            all_embeddings.extend(e.embedding for e in ordered)
            logger.debug("Embedded batch %d/%d", i // _BATCH_SIZE + 1, -(-len(texts) // _BATCH_SIZE))

        return np.array(all_embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single string. Returns shape (dim,)."""
        return self.embed([text])[0]

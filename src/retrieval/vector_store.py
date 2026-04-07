import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store using inner-product search on L2-normalised vectors.

    Normalising before insertion means IndexFlatIP computes cosine similarity,
    which is what we want for semantic retrieval.
    """

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add embeddings and their associated metadata records."""
        if len(embeddings) != len(metadata):
            raise ValueError("embeddings and metadata must have the same length.")
        vecs = embeddings.astype(np.float32).copy()
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(metadata)
        logger.info("Added %d vectors (total: %d)", len(embeddings), self.index.ntotal)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[float, dict]]:
        """Return (cosine_score, metadata) pairs for the top_k nearest neighbours."""
        vec = query_embedding.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
        return [
            (float(score), self.metadata[idx])
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, metadata_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info("Saved index to %s (%d vectors)", index_path, self.index.ntotal)

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "VectorStore":
        index = faiss.read_index(index_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        store = cls.__new__(cls)
        store.index = index
        store.metadata = metadata
        logger.info("Loaded index from %s (%d vectors)", index_path, index.ntotal)
        return store

    def __len__(self) -> int:
        return self.index.ntotal

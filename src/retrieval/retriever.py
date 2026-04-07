import logging

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import Settings
from src.ingestion.dataClasses import Chunk
from src.ingestion.embedding import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 keyword search with dense semantic search, then reranks.

    Pipeline:
      1. Semantic search  — top_k results from FAISS cosine similarity
      2. BM25 search      — top_k results from sparse keyword matching
      3. RRF fusion       — merge both ranked lists via Reciprocal Rank Fusion
      4. Cross-encoder    — rerank the fused candidates for final ordering

    alpha controls the hybrid balance: 0.0 = BM25 only, 1.0 = semantic only.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: list[Chunk],
        embedder: Embedder,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.chunks = chunks
        self.embedder = embedder
        self.settings = settings

        logger.info("Building BM25 index over %d chunks...", len(chunks))
        tokenised = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenised)

        logger.info("Loading cross-encoder '%s'...", settings.reranker_model)
        self.reranker = CrossEncoder(settings.reranker_model)

        logger.info("HybridRetriever ready.")

    def retrieve(self, query: str) -> list[tuple[float, Chunk]]:
        """Return (rerank_score, chunk) pairs, best first, capped at rerank_top_k."""
        top_k = self.settings.top_k
        alpha = self.settings.alpha

        # 1. Semantic search
        q_emb = self.embedder.embed_one(query)
        sem_results = self.vector_store.search(q_emb, top_k)  # [(score, metadata)]

        # 2. BM25 search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()

        # 3. Reciprocal Rank Fusion (RRF) — k=60 is a common default
        rrf: dict[int, float] = {}
        k = 60

        for rank, (_, meta) in enumerate(sem_results):
            cid = meta["chunk_id"]
            rrf[cid] = rrf.get(cid, 0.0) + alpha / (rank + k)

        for rank, idx in enumerate(top_bm25_idx):
            rrf[idx] = rrf.get(idx, 0.0) + (1.0 - alpha) / (rank + k)

        candidate_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)[:top_k]
        candidates = [self.chunks[cid] for cid in candidate_ids]

        logger.debug(
            "RRF produced %d candidates from semantic+BM25 (%d results each)",
            len(candidates), top_k,
        )

        # 4. Cross-encoder reranking
        pairs = [(query, c.text) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)

        final = [(float(s), c) for s, c in ranked[: self.settings.rerank_top_k]]
        logger.info(
            "Retrieved %d chunks (top score: %.4f)",
            len(final), final[0][0] if final else 0.0,
        )
        return final

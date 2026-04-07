import json
import logging
from pathlib import Path

from config import Settings
from src.generation.generation import generate_answer
from src.ingestion.dataClasses import Chunk
from src.ingestion.embedding import Embedder
from src.retrieval.retriever import HybridRetriever
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_retriever(settings: Settings) -> HybridRetriever:
    """Reconstruct a HybridRetriever from persisted index + chunks.

    Raises FileNotFoundError if ingest has not been run yet.
    """
    if not Path(settings.index_path).exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{settings.index_path}'. Run 'ingest' first."
        )
    if not Path(settings.chunks_path).exists():
        raise FileNotFoundError(
            f"No chunks file found at '{settings.chunks_path}'. Run 'ingest' first."
        )

    meta_path = settings.index_path.replace(".faiss", "_meta.json")
    store = VectorStore.load(settings.index_path, meta_path)

    with open(settings.chunks_path, encoding="utf-8") as f:
        raw = json.load(f)
    chunks = [Chunk(text=r["text"], metadata=r["metadata"]) for r in raw]

    embedder = Embedder(settings)
    return HybridRetriever(store, chunks, embedder, settings)


def query(question: str, retriever: HybridRetriever, settings: Settings) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    Returns a dict with 'answer' and 'sources' so callers can display
    provenance alongside the answer.
    """
    logger.info("Query: %s", question)

    results = retriever.retrieve(question)
    chunks = [c for _, c in results]
    scores = [s for s, _ in results]

    logger.info(
        "Top-%d chunks retrieved (scores: %s)",
        len(chunks), [f"{s:.4f}" for s in scores],
    )

    answer = generate_answer(question, chunks, settings)

    return {
        "answer": answer,
        "sources": [
            {
                "score": round(score, 4),
                "section": chunk.metadata.get("section", ""),
                "subsection": chunk.metadata.get("subsection", ""),
                "source": chunk.metadata.get("source", ""),
                "text_preview": chunk.text[:200],
            }
            for score, chunk in zip(scores, chunks)
        ],
    }

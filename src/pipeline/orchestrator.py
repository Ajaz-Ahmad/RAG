import json
import logging
from pathlib import Path

from config import Settings
from src.ingestion.chunker import merge_wiki_chunks
from src.ingestion.dataClasses import Chunk
from src.ingestion.embedding import Embedder
from src.ingestion.loader import load_data_from_url
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def ingest(urls: list[str], settings: Settings) -> tuple[VectorStore, list[Chunk]]:
    """Load, chunk, embed, and index one or more Wikipedia URLs.

    Saves the FAISS index and serialised chunks to disk so query_pipeline
    can reload them without re-embedding.
    """
    all_chunks: list[Chunk] = []

    for url in urls:
        wiki_chunks = load_data_from_url(url)
        chunks = merge_wiki_chunks(wiki_chunks, settings.max_words_per_chunk)
        # Tag every chunk with its source URL so we can trace answers back
        for chunk in chunks:
            chunk.metadata["source"] = url
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks produced from the provided URLs.")

    # Assign stable integer IDs — the retriever uses these as BM25 list indices
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = i

    logger.info("Total chunks to embed: %d", len(all_chunks))

    embedder = Embedder(settings)
    texts = [c.text for c in all_chunks]
    embeddings = embedder.embed(texts)

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, [c.metadata for c in all_chunks])

    # Persist index + metadata sidecar
    meta_path = settings.index_path.replace(".faiss", "_meta.json")
    store.save(settings.index_path, meta_path)

    # Persist full chunk text separately for reconstruction at query time
    Path(settings.chunks_path).parent.mkdir(parents=True, exist_ok=True)
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"text": c.text, "metadata": c.metadata} for c in all_chunks],
            f, ensure_ascii=False, indent=2,
        )

    logger.info("Ingestion complete. Index: %s  Chunks: %s", settings.index_path, settings.chunks_path)
    return store, all_chunks

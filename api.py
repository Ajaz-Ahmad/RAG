"""FastAPI entry point for the Wikipedia RAG pipeline.

Exposes two routes:
  POST /ingest  — scrape a Wikipedia URL, chunk, embed, hold in memory
  POST /query   — retrieve relevant chunks and generate a grounded answer

State is kept in a process-level dict keyed by the article URL.
This is intentional for a single-instance demo deployment.
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from config import settings
from src.generation.generation import generate_answer
from src.ingestion.chunker import merge_wiki_chunks
from src.ingestion.embedding import Embedder
from src.ingestion.loader import load_data_from_url
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-level session store
# key   = Wikipedia URL (used as session identifier)
# value = { store, chunks, bm25, title }
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings)
    return _embedder


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up embedding model…")
    get_embedder()
    logger.info("Embedding model ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Wikipedia RAG API",
    description=(
        "Ingest any English Wikipedia article, then ask questions grounded in it.\n\n"
        "**Workflow:**\n"
        "1. `POST /ingest` — supply a Wikipedia URL\n"
        "2. `POST /query`  — ask a question, get a cited answer\n\n"
        "Retrieval uses BM25 + FAISS semantic search with Reciprocal Rank Fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    title: str
    session_key: str
    chunks_count: int


class QueryRequest(BaseModel):
    question: str
    session_key: str
    top_k: int = 5
    alpha: float = 0.5  # 0 = BM25 only, 1 = semantic only


class SourceOut(BaseModel):
    section: str
    subsection: str
    preview: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest(req: IngestRequest):
    """Scrape a Wikipedia article, chunk it, embed it, and store it in memory."""
    if "en.wikipedia.org/wiki/" not in req.url:
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid English Wikipedia URL, e.g. https://en.wikipedia.org/wiki/Diabetes",
        )

    try:
        raw_chunks = load_data_from_url(req.url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Wikipedia article: {exc}")

    if not raw_chunks:
        raise HTTPException(status_code=422, detail="No content extracted from this article.")

    page_title = raw_chunks[0].page_title
    chunks = merge_wiki_chunks(raw_chunks, settings.max_words_per_chunk)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = req.url
        chunk.metadata.setdefault("page_title", page_title)

    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, [c.metadata for c in chunks])

    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    _sessions[req.url] = {"store": store, "chunks": chunks, "bm25": bm25, "title": page_title}

    logger.info("Ingested '%s' → %d chunks  (session: %s)", page_title, len(chunks), req.url)
    return IngestResponse(title=page_title, session_key=req.url, chunks_count=len(chunks))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """Retrieve relevant chunks and generate a grounded answer."""
    session = _sessions.get(req.session_key)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /ingest with the Wikipedia URL first.",
        )

    store: VectorStore = session["store"]
    chunks = session["chunks"]
    bm25: BM25Okapi = session["bm25"]
    embedder = get_embedder()

    top_k = min(req.top_k, len(chunks))
    alpha = max(0.0, min(1.0, req.alpha))
    k = 60  # RRF constant

    # 1. Semantic search
    q_emb = embedder.embed_one(req.question)
    sem_results = store.search(q_emb, top_k)  # [(score, metadata)]

    # 2. BM25 search
    bm25_scores = bm25.get_scores(req.question.lower().split())
    top_bm25_idx: list[int] = np.argsort(bm25_scores)[::-1][:top_k].tolist()

    # 3. Reciprocal Rank Fusion
    rrf: dict[int, float] = {}
    for rank, (_, meta) in enumerate(sem_results):
        cid = int(meta.get("chunk_id", 0))
        rrf[cid] = rrf.get(cid, 0.0) + alpha / (rank + k)
    for rank, idx in enumerate(top_bm25_idx):
        rrf[idx] = rrf.get(idx, 0.0) + (1.0 - alpha) / (rank + k)

    candidate_ids = sorted(rrf, key=rrf.__getitem__, reverse=True)
    top_chunks = [chunks[cid] for cid in candidate_ids[:4]]

    # 4. Generate answer
    answer = generate_answer(req.question, top_chunks, settings)

    sources = [
        SourceOut(
            section=c.metadata.get("section", ""),
            subsection=c.metadata.get("subsection", ""),
            preview=c.text[:160] + ("…" if len(c.text) > 160 else ""),
            score=round(rrf.get(candidate_ids[i], 0.0), 4),
        )
        for i, c in enumerate(top_chunks)
    ]

    return QueryResponse(question=req.question, answer=answer, sources=sources)

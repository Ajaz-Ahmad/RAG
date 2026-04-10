---
title: Wikipedia RAG API
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Wikipedia RAG API

A retrieval-augmented generation pipeline that lets you ask questions grounded in any Wikipedia article.

## How it works

1. `POST /ingest` — supply a Wikipedia URL; the article is scraped, chunked, embedded, and held in memory
2. `POST /query`  — ask a question; BM25 + FAISS hybrid retrieval selects the best passages, GPT-4o-mini generates a cited answer

## Retrieval pipeline

- **Loader**: scrapes Wikipedia with section-aware paragraph extraction
- **Chunker**: merges paragraphs within the same section, capped at 200 words
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, free, no API key)
- **Vector store**: FAISS with cosine similarity
- **Hybrid retrieval**: BM25 + semantic search fused via Reciprocal Rank Fusion
- **Generation**: GPT-4o-mini (requires `OPENAI_API_KEY`)

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | **Groq key** (free at console.groq.com) — starts with `gsk_` |
| `OPENAI_BASE_URL` | No | Defaults to `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | No | Defaults to `llama-3.3-70b-versatile` |

## CLI usage (local)

```bash
pip install -r requirements-api.txt

# Start the API
uvicorn api:app --reload

# Or use the original CLI
python main.py ingest https://en.wikipedia.org/wiki/Diabetes
python main.py query "What causes type 2 diabetes?"
```

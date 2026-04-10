from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embeddings — local sentence-transformers by default (free, no API key needed).
    # Set USE_LOCAL_EMBEDDINGS=false + OPENAI_API_KEY to use OpenAI instead.
    use_local_embeddings: bool = True
    local_embedding_model: str = "all-MiniLM-L6-v2"   # 80 MB, runs fast on CPU

    # OpenAI-compatible embedding API (used when use_local_embeddings=false)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model: str = "text-embedding-3-small"

    # Generation — Groq by default (free, no credit card).
    # Override OPENAI_BASE_URL + OPENAI_MODEL to use any OpenAI-compatible API.
    openai_api_key: str = "EMPTY"                        # set via OPENAI_API_KEY (Groq or OpenAI key)
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_model: str = "llama-3.3-70b-versatile"        # free on Groq; swap for gpt-4o-mini if using OpenAI
    temperature: float = 0.2
    max_tokens: int = 512

    # Chunking
    max_words_per_chunk: int = 200

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 3
    alpha: float = 0.5       # 0 = BM25 only, 1 = semantic only

    # Reranker (cross-encoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Persistence
    index_path: str = "data/index.faiss"
    chunks_path: str = "data/chunks.json"

    def __init__(self, **data):
        super().__init__(**data)
        if self.embedding_api_key == "EMPTY" and self.openai_api_key != "EMPTY":
            self.embedding_api_key = self.openai_api_key

    class Config:
        env_file = ".env"


settings = Settings()

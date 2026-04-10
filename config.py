from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding — defaults to OpenAI; override with EMBEDDING_BASE_URL for vLLM
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = "EMPTY"       # set via EMBEDDING_API_KEY or falls back to openai_api_key
    embedding_model: str = "text-embedding-3-small"

    # Generation — OpenAI or any compatible endpoint
    openai_api_key: str = "EMPTY"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512

    # Chunking
    max_words_per_chunk: int = 200

    # Retrieval
    top_k: int = 10          # candidates fetched before reranking
    rerank_top_k: int = 3    # final chunks sent to the LLM
    alpha: float = 0.5       # hybrid weight: 0 = BM25 only, 1 = semantic only

    # Reranker (cross-encoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Persistence
    index_path: str = "data/index.faiss"
    chunks_path: str = "data/chunks.json"

    def __init__(self, **data):
        super().__init__(**data)
        # If no dedicated embedding key is set, reuse the OpenAI key.
        if self.embedding_api_key == "EMPTY" and self.openai_api_key != "EMPTY":
            self.embedding_api_key = self.openai_api_key

    class Config:
        env_file = ".env"


settings = Settings()

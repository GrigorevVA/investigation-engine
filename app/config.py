"""All settings in one place. Values come from environment variables or .env."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM: any OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter...)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 1.0

    # Vector store: empty QDRANT_URL means local on-disk Qdrant in qdrant_path
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = "data/qdrant"
    collection: str = "documents_paragraph"

    # Embeddings run locally, documents never leave the machine for this step
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking: "paragraph" (strategy B) or "fixed" (strategy A), size in characters
    chunk_strategy: str = "paragraph"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    search_mode: str = "hybrid"  # "vector" | "bm25" | "hybrid"
    top_k: int = 8
    candidate_k: int = 30  # how many candidates each retriever gives to the fusion
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Investigation
    max_subquestions: int = 3
    max_context_chunks: int = 12
    max_tool_rounds: int = 2


@lru_cache
def get_settings() -> Settings:
    return Settings()

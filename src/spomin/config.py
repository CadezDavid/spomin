from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Local archive
    database_path: Path = Path("./data/spomin.db")

    # Embedding model for semantic search
    embedder_base_url: str = "http://host.docker.internal:8081/v1"
    embedder_model: str = "embeddinggemma-300M-Q8_0.gguf"
    embedder_api_key: str = "dummy"

    # Chunking and background embedding
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800
    embedding_batch_size: int = 16
    embedding_poll_interval: float = 2.0
    embedding_retry_interval: float = 15.0

    # Retrieval ranking
    max_search_limit: int = 50
    keyword_weight: float = 1.0
    vector_weight: float = 1.0
    recency_weight: float = 0.15
    access_weight: float = 0.05
    core_memory_boost: float = 0.25
    recency_half_life_days: float = 30.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8084

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

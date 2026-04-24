from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Neo4j connection
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # LLM for entity extraction (separate from chat LLM)
    extraction_llm_base_url: str = "http://host.docker.internal:8082/v1"
    extraction_llm_model: str = "Qwen3-0.5B-Q4_K_M.gguf"
    extraction_llm_api_key: str = "dummy"

    # Embedding model for semantic search
    embedder_base_url: str = "http://host.docker.internal:8081/v1"
    embedder_model: str = "embeddinggemma-300M-Q8_0.gguf"
    embedder_api_key: str = "dummy"

    # Server
    host: str = "0.0.0.0"
    port: int = 8084
    user_id: str = "default"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

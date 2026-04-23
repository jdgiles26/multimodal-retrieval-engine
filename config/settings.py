"""
Multimodal Retrieval System Configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration"""

    # DashScope API
    DASHSCOPE_API_KEY: Optional[str] = None

    # Embedding model configuration
    EMBEDDING_MODEL: str = "qwen3-vl-embedding"
    EMBEDDING_DIMENSION: int = 2560

    # Rerank model configuration
    RERANK_MODEL: str = "qwen3-rerank"

    # Milvus configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "duckyo_travel_memories"

    # MinIO configuration (image storage)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: Optional[str] = None
    MINIO_SECRET_KEY: Optional[str] = None
    MINIO_BUCKET: str = "duckyo-memories"
    MINIO_SECURE: bool = False

    # Data directory for local images
    DATA_DIR: str = "./data"

    # Retrieval configuration
    DEFAULT_TOP_K: int = 10
    RERANK_TOP_K: int = 100

    # API configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Langfuse configuration (optional)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Lazy singleton to avoid import-time crashes
_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


# Backward-compatible module-level export (lazy)
settings = get_settings()

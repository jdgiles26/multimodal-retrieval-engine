"""
Duckyo 多模态检索系统配置
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # DashScope API
    DASHSCOPE_API_KEY: str
    
    # Embedding 模型配置
    EMBEDDING_MODEL: str = "qwen3-vl-embedding"
    EMBEDDING_DIMENSION: int = 2560  # Qwen3-VL 默认维度
    
    # Rerank 模型配置
    RERANK_MODEL: str = "qwen3-rerank"
    
    # Milvus 配置
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "duckyo_travel_memories"
    
    # MinIO 配置 (图片存储)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "duckyo-memories"
    MINIO_SECURE: bool = False
    
    # 检索配置
    DEFAULT_TOP_K: int = 10
    RERANK_TOP_K: int = 100  # 粗筛数量
    
    # API 配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Langfuse 配置 (可选)
    # 在 https://cloud.langfuse.com 注册并获取 API Keys
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_ENABLED: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()

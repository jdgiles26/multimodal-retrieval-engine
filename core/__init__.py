"""
Core module exports - LlamaIndex version
"""
from core.llamaindex_engine import (
    LlamaIndexMultimodalRetriever,
    get_llamaindex_retriever,
    Qwen3VLEmbedding,
    DashScopeReranker,
    Qwen36Summarizer,
)

# Compatibility aliases
MultimodalRetrievalEngine = LlamaIndexMultimodalRetriever
get_engine = get_llamaindex_retriever
Qwen3VLReranker = DashScopeReranker

__all__ = [
    "LlamaIndexMultimodalRetriever",
    "get_llamaindex_retriever",
    "MultimodalRetrievalEngine",
    "get_engine",
    "Qwen3VLEmbedding",
    "DashScopeReranker",
    "Qwen3VLReranker",
    "Qwen36Summarizer",
]

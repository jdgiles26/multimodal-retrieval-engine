"""
Core 模块导出 - LlamaIndex 版本 (精简版)
"""
from core.llamaindex_engine import LlamaIndexMultimodalRetriever, get_llamaindex_retriever

# 主引擎
MultimodalRetrievalEngine = LlamaIndexMultimodalRetriever
get_engine = get_llamaindex_retriever

__all__ = [
    "LlamaIndexMultimodalRetriever",
    "get_llamaindex_retriever",
    "MultimodalRetrievalEngine",
    "get_engine",
]
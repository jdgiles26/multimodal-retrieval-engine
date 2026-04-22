"""
LlamaIndex 多模态检索系统 - 完整实现
使用 Qwen3-VL Embedding + Milvus
"""
from typing import List, Dict, Any, Optional
import dashscope
import numpy as np
from loguru import logger
import os
import time

from llama_index.core import (
    Document, VectorStoreIndex, StorageContext,
    Settings, QueryBundle, Settings
)
from llama_index.core.schema import ImageDocument, TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore


class Qwen3VLEmbedding(BaseEmbedding):
    """
    LlamaIndex 兼容的 Qwen3-VL Embedding
    
    支持:
    - 文本编码
    - 图片编码
    - 图文融合编码
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 2560,
        **kwargs
    ):
        # 先设置实例变量
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        
        # 设置 dashscope
        dashscope.api_key = api_key
        
        # 调用父类初始化 (不传 model)
        super().__init__(**kwargs)
        
        logger.info(f"Qwen3VLEmbedding initialized: model={model}, dim={dimension}")
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """文本编码"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self._model,
            input=[{"text": text}],
            dimension=self._dimension
        )
        if resp.status_code != 200:
            raise Exception(f"Embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
    
    def _get_image_embedding(self, image_url: str) -> List[float]:
        """图片编码"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self._model,
            input=[{"image": image_url}],
            dimension=self._dimension
        )
        if resp.status_code != 200:
            raise Exception(f"Image embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """查询编码"""
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步文本编码"""
        return self._get_text_embedding(text)
    
    async def _aget_image_embedding(self, image_url: str) -> List[float]:
        """异步图片编码"""
        return self._get_image_embedding(image_url)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步查询编码"""
        return self._get_query_embedding(query)


class LlamaIndexMultimodalRetriever:
    """
    LlamaIndex 多模态检索引擎
    """
    
    def __init__(
        self,
        api_key: str,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "duckyo_travel_memories",
        embedding_model: str = "qwen3-vl-embedding",
        embedding_dim: int = 2560
    ):
        self.api_key = api_key
        
        # 初始化 Embedding
        self.embed_model = Qwen3VLEmbedding(
            api_key=api_key,
            model=embedding_model,
            dimension=embedding_dim
        )
        
        # 设置全局配置
        Settings.embed_model = self.embed_model
        
        # 初始化 Milvus
        self.vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=collection_name,
            dim=embedding_dim,
            index_config={
                "index_type": "HNSW",
                "metric_type": "COSINE"
            }
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        self.index = None
        logger.info(f"LlamaIndexMultimodalRetriever initialized: collection={collection_name}")
    
    def create_index(self, documents: List[Document]):
        """
        创建索引
        """
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        logger.info(f"Created index with {len(documents)} documents")
    
    def add_travel_memories(
        self,
        memories: List[Dict[str, Any]]
    ):
        """
        添加旅途记忆
        
        Args:
            memories: 记忆列表，每项包含:
                - content: 文本描述
                - city: 城市
                - country: 国家
                - date: 日期
                - image_url: 图片URL (可选)
                - tags: 标签列表
                - mood: 心情
                - outfit: 服装
                - source: 来源
        """
        documents = []
        for mem in memories:
            # 如果有图片URL，创建多模态文档
            if mem.get("image_url"):
                doc = ImageDocument(
                    text=mem.get("content", ""),
                    image_url=mem.get("image_url"),
                    metadata={
                        "city": mem.get("city", ""),
                        "country": mem.get("country", ""),
                        "date": mem.get("date", ""),
                        "tags": mem.get("tags", []),
                        "mood": mem.get("mood", ""),
                        "outfit": mem.get("outfit", ""),
                        "source": mem.get("source", "")
                    }
                )
            else:
                doc = Document(
                    text=mem.get("content", ""),
                    metadata={
                        "city": mem.get("city", ""),
                        "country": mem.get("country", ""),
                        "date": mem.get("date", ""),
                        "tags": mem.get("tags", []),
                        "mood": mem.get("mood", ""),
                        "outfit": mem.get("outfit", ""),
                        "source": mem.get("source", "")
                    }
                )
            documents.append(doc)
        
        if self.index is None:
            self.create_index(documents)
        else:
            for doc in documents:
                self.index.insert(doc)
        
        logger.info(f"Added {len(documents)} memories")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        搜索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件 (city, date_from, date_to, tags)
        
        Returns:
            检索结果列表
        """
        if self.index is None:
            raise ValueError("Index not created. Please add documents first.")
        
        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        # 检索
        nodes = retriever.retrieve(query)
        
        # 格式化结果
        results = []
        for node in nodes:
            results.append({
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
                "id": node.node.id_
            })
        
        return results
    
    def search_by_image(
        self,
        image_url: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        图搜图
        
        Args:
            image_url: 图片URL
            top_k: 返回数量
        
        Returns:
            检索结果列表
        """
        if self.index is None:
            raise ValueError("Index not created.")
        
        # 编码图片
        query_embedding = self.embed_model._get_image_embedding(image_url)
        
        # 创建检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        # 构建查询bundle
        query_bundle = QueryBundle(
            query_str="",  # 空文本
            embedding=query_embedding
        )
        
        nodes = retriever.retrieve(query_bundle)
        
        results = []
        for node in nodes:
            results.append({
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
                "id": node.node.id_
            })
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "collection": self.vector_store.collection_name,
            "count": self.vector_store.query().get("count", 0)
        }


# ========== 便捷函数 ==========

def create_retriever(
    api_key: Optional[str] = None,
    milvus_uri: str = "http://localhost:19530",
    collection_name: str = "duckyo_travel_memories"
) -> LlamaIndexMultimodalRetriever:
    """
    创建检索器
    
    Args:
        api_key: DashScope API Key
        milvus_uri: Milvus 地址
        collection_name: Collection 名称
    
    Returns:
        LlamaIndexMultimodalRetriever 实例
    """
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
    
    return LlamaIndexMultimodalRetriever(
        api_key=api_key,
        milvus_uri=milvus_uri,
        collection_name=collection_name
    )


# ========== CLI 工具 ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LlamaIndex 多模态检索系统")
    parser.add_argument("--action", choices=["init", "add", "search", "search-image", "stats"], required=True)
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("--image", type=str, help="图片URL")
    parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    parser.add_argument("--data", type=str, help="数据JSON文件")
    
    args = parser.parse_args()
    
    # 创建检索器
    retriever = create_retriever()
    
    if args.action == "init":
        print("🚀 初始化 Collection...")
        retriever.create_index([])
        print("✅ 初始化完成!")
    
    elif args.action == "add":
        if not args.data:
            print("❌ 请指定 --data 参数")
            exit(1)
        
        import json
        with open(args.data, 'r') as f:
            memories = json.load(f)
        
        retriever.add_travel_memories(memories)
        print(f"✅ 添加了 {len(memories)} 条记忆!")
    
    elif args.action == "search":
        if not args.query:
            print("❌ 请指定 --query 参数")
            exit(1)
        
        results = retriever.search(args.query, top_k=args.top_k)
        
        print(f"\n🔍 搜索: '{args.query}'")
        print(f"📊 找到 {len(results)} 条结果\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] 分数: {r['score']:.3f}")
            print(f"    内容: {r['content'][:50]}...")
            print(f"    城市: {r['metadata'].get('city', 'N/A')}")
            print(f"    日期: {r['metadata'].get('date', 'N/A')}")
            print()
    
    elif args.action == "search-image":
        if not args.image:
            print("❌ 请指定 --image 参数")
            exit(1)
        
        results = retriever.search_by_image(args.image, top_k=args.top_k)
        
        print(f"\n🖼️  图搜图: {args.image}")
        print(f"📊 找到 {len(results)} 条结果\n")
        
        for i, r in enumerate(results, 1):
            print(f"[{i}] 分数: {r['score']:.3f}")
            print(f"    内容: {r['content'][:50]}...")
            print(f"    城市: {r['metadata'].get('city', 'N/A')}")
            print()
    
    elif args.action == "stats":
        stats = retriever.get_stats()
        print("\n📊 Collection 统计:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
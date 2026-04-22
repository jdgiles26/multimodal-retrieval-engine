"""
LlamaIndex 多模态检索引擎
使用 Qwen3-VL Embedding + Milvus + Langfuse 观测
"""
from typing import List, Dict, Any, Optional
import dashscope
import numpy as np
from loguru import logger

from llama_index.core import (
    Document, VectorStoreIndex, StorageContext,
    Settings
)

# Langfuse 集成
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not installed. Run: pip install langfuse")
from llama_index.core.schema import ImageDocument
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle


class Qwen36Summarizer:
    """
    搜索结果总结器
    使用 qwen3.6-plus 生成检索结果的智能总结
    """
    
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        self.api_key = api_key
        self.model = model
        logger.info(f"Qwen36Summarizer initialized: model={model}")
    
    def summarize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_length: int = 500
    ) -> str:
        """
        生成搜索结果的总结
        
        Args:
            query: 用户查询
            results: 检索结果列表
            max_length: 总结最大长度
            
        Returns:
            总结文本
        """
        if not results:
            return "未找到相关结果。"
        
        # 构建结果摘要
        results_text = []
        for i, r in enumerate(results[:10], 1):  # 最多取前10条
            content = r.get('content', '')[:200]  # 截断内容
            city = r.get('city', '未知地点')
            score = r.get('score', 0)
            results_text.append(f"[{i}] {city}: {content}... (相关度: {score:.2f})")
        
        results_str = "\n".join(results_text)
        
        # 构建 prompt
        prompt = f"""基于以下搜索结果，为用户查询生成一个简洁的总结。

用户查询: {query}

搜索结果:
{results_str}

请生成一个简洁的总结，包含:
1. 找到的主要内容
2. 涉及的地点
3. 整体氛围或主题

总结:"""
        
        try:
            # qwen3.6-plus 是多模态模型，需要使用 MultiModalConversation
            # 临时设置 API key
            original_key = dashscope.api_key
            dashscope.api_key = self.api_key
            
            resp = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": [{"text": "你是一个智能助手，擅长总结搜索结果。"}]},
                    {"role": "user", "content": [{"text": prompt}]}
                ]
            )
            
            # 恢复原始 API key
            dashscope.api_key = original_key
            
            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                # 提取文本内容
                if isinstance(content, list) and len(content) > 0:
                    summary = content[0].get('text', '')
                else:
                    summary = str(content)
                return summary[:max_length]
            else:
                logger.warning(f"Summarization failed: {resp.message}")
                return self._fallback_summary(results)
                
        except Exception as e:
            logger.warning(f"Summarization error: {e}")
            return self._fallback_summary(results)
    
    def _fallback_summary(self, results: List[Dict[str, Any]]) -> str:
        """备用总结"""
        cities = set()
        for r in results:
            city = r.get('city')
            if city:
                cities.add(city)
        
        city_str = ", ".join(cities) if cities else "多个地点"
        return f"找到 {len(results)} 条相关记忆，涉及地点: {city_str}。"
    
    async def asummarize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_length: int = 500
    ) -> str:
        """异步版本"""
        return self.summarize_results(query, results, max_length)


class Qwen3VLEmbedding(BaseEmbedding):
    """
    LlamaIndex 兼容的 Qwen3-VL Embedding
    """
    
    model: str = "qwen3-vl-embedding"
    dimension: int = 2560
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-vl-embedding",
        dimension: int = 2560,
        **kwargs
    ):
        # 先设置属性再调用 super
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'dimension', dimension)
        super().__init__(**kwargs)
        dashscope.api_key = api_key
        logger.info(f"Qwen3VLEmbedding initialized: model={model}, dim={dimension}")
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """文本编码"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=[{"text": text}],
            dimension=self.dimension
        )
        if resp.status_code != 200:
            raise Exception(f"Embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
    
    def _get_image_embedding(self, image_url: str) -> List[float]:
        """图片编码"""
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=[{"image": image_url}],
            dimension=self.dimension
        )
        if resp.status_code != 200:
            raise Exception(f"Image embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
    
    def _get_multimodal_embedding(self, text: str, image: str) -> List[float]:
        """多模态编码（文本+图片），与入库时保持一致"""
        import dashscope
        
        # 构建多模态输入
        input_list = []
        if text:
            input_list.append({"text": text})
        if image:
            input_list.append({"image": image})
        
        if not input_list:
            raise Exception("At least one of text or image must be provided")
        
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=input_list,
            dimension=self.dimension
        )
        if resp.status_code != 200:
            raise Exception(f"Multimodal embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]
    
    def _get_image_base64(self, image_url: str) -> Optional[str]:
        """将图片 URL 转换为本地 base64"""
        import base64
        from pathlib import Path
        import os
        
        if not image_url:
            return None
        
        # 提取文件名
        filename = image_url.split('/')[-1]
        local_path = f"/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval/data/xhs_images/{filename}"
        
        # 尝试不同后缀
        if not os.path.exists(local_path):
            base_name = filename.replace('.png', '').replace('.jpg', '')
            for ext in ['.png', '.jpg']:
                test_path = f"/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval/data/xhs_images/{base_name}{ext}"
                if os.path.exists(test_path):
                    local_path = test_path
                    break
        
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        return None
    
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


class DashScopeReranker(BaseNodePostprocessor):
    """
    LlamaIndex 兼容的 DashScope Reranker
    支持 qwen3-rerank 多模态重排序
    """
    
    model: str = "qwen3-rerank"
    top_n: int = 10
    _api_key: str = ""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        top_n: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        # 使用单下划线避免 Pydantic 字段冲突
        self._api_key = api_key
        self.model = model
        self.top_n = top_n
        self._query_image = None  # 用于存储图搜图的图片
        dashscope.api_key = api_key
        logger.info(f"DashScopeReranker initialized: model={model}, top_n={top_n}")
    
    def _clean_text_for_api(self, text: str) -> str:
        """清理文本，移除 emoji 表情符号，保留中文、英文和基本标点，用于 API 调用"""
        import re
        if not text:
            return "query"
        
        # 只移除 emoji 表情符号（特定范围）
        # 使用更精确的模式匹配常见的 emoji
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # emoticons
            (0x1F300, 0x1F5FF),  # symbols & pictographs
            (0x1F680, 0x1F6FF),  # transport & map
            (0x1F1E0, 0x1F1FF),  # flags
            (0x2600, 0x26FF),    # misc symbols
            (0x2700, 0x27BF),    # dingbats
        ]
        
        result = []
        for char in text:
            code = ord(char)
            is_emoji = any(start <= code <= end for start, end in emoji_ranges)
            if not is_emoji:
                result.append(char)
        
        cleaned = ''.join(result).strip()
        
        # 如果清理后为空，返回原始文本（截断）
        if not cleaned:
            return text[:500] if len(text) <= 500 else text[:500]
        
        # 限制长度
        return cleaned[:500]
    
    def _call_rerank_api(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_image: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """
        调用 DashScope Rerank API
        支持多模态输入 (qwen3-rerank) 和纯文本输入
        
        Args:
            query: 文本查询
            documents: 文档列表
            query_image: 可选的图片 base64 或 URL
        """
        try:
            # 设置 API Key
            api_key = self._api_key if hasattr(self, '_api_key') else getattr(self, 'api_key', '')
            dashscope.api_key = api_key
            
            # 清理查询文本
            clean_query = self._clean_text_for_api(query)
            
            # 构建 query
            if query_image:
                # 图搜图：使用图片作为 query
                query_input = {"image": query_image}
            else:
                # 文本搜索：使用文本作为 query
                query_input = {"text": clean_query}
            
            # 使用 DashScope SDK 调用
            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query_input,
                documents=documents,
                top_n=min(self.top_n, len(documents))
            )
            
            if resp.status_code == 200:
                return resp.output.get("results", [])
            else:
                logger.warning(f"Rerank API failed: {resp.message}")
                return None
                
        except Exception as e:
            logger.warning(f"Rerank API call failed: {e}")
            return None
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """对节点进行重排序"""
        if not nodes or not query_bundle:
            return nodes
        
        query = query_bundle.query_str
        
        # 构建多模态文档列表
        # 注意：qwen3-rerank 不支持 text+image 混合，每个文档只能用一种类型
        # 如果有 image_url，尝试使用图片格式；否则使用文本格式
        documents = []
        for node in nodes:
            if hasattr(node, 'node'):
                node_obj = node.node
                text = node_obj.text if hasattr(node_obj, 'text') else str(node_obj)
                metadata = node_obj.metadata if hasattr(node_obj, 'metadata') else {}
            else:
                text = node.text if hasattr(node, 'text') else str(node)
                metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            image_url = metadata.get("image_url", "")
            
            # 选择格式：有图片且是公开可访问的 URL 才用图片格式
            # localhost 或私有地址使用文本格式
            if image_url and image_url.startswith('http') and not (
                'localhost' in image_url or 
                '127.0.0.1' in image_url or
                '192.168.' in image_url or
                '10.' in image_url
            ):
                # 使用图片格式（需要公开可访问的 URL）
                documents.append({"image": image_url})
            else:
                # 使用文本格式（localhost 或没有图片）
                # 清理文本中的 emoji，避免 API 错误
                clean_text = self._clean_text_for_api(text) if text else ""
                documents.append({"text": clean_text})
        
        try:
            # 调用多模态 rerank API，如果有 query_image 则传递
            results = self._call_rerank_api(query, documents, self._query_image)
            
            # 清空 query_image，避免影响下次调用
            self._query_image = None
            
            if results is None:
                logger.warning("Rerank API returned None, falling back to vector scores")
                return nodes[:self.top_n]
            
            if not results:
                return nodes[:self.top_n]
            
            # 根据 index 重新排序
            reranked = []
            for r in results:
                idx = r.get("index", 0)
                if 0 <= idx < len(nodes):
                    # 更新分数
                    nodes[idx].score = r.get("relevance_score", nodes[idx].score)
                    reranked.append(nodes[idx])
            
            logger.info(f"Reranked {len(nodes)} nodes to {len(reranked)} using {self.model}")
            return reranked
            
        except Exception as e:
            logger.warning(f"Rerank error: {e}")
            return nodes[:self.top_n]


from config.settings import settings


# 导出别名保持兼容
Qwen3VLReranker = DashScopeReranker


# 导出 Summarizer
__all__ = [
    "Qwen3VLEmbedding",
    "DashScopeReranker",
    "Qwen36Summarizer",
    "LlamaIndexMultimodalRetriever",
    "get_llamaindex_retriever",
    "Qwen3VLReranker",
]


class LlamaIndexMultimodalRetriever:
    """
    LlamaIndex 多模态检索引擎
    兼容自定义引擎的 API 接口
    集成 Langfuse 观测
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "duckyo_travel_memories",
        embedding_model: str = "qwen3-vl-embedding",
        embedding_dim: int = 2560,
        rerank_model: str = "qwen3-rerank",
        rerank_top_n: int = 10,
        summarizer_model: str = "qwen3.6-plus",
        enable_summary: bool = True,
        enable_langfuse: bool = None
    ):
        self.api_key = api_key or settings.DASHSCOPE_API_KEY
        self.rerank_top_n = rerank_top_n
        self.enable_summary = enable_summary
        
        # 初始化 Langfuse
        self.langfuse = None
        if enable_langfuse is None:
            enable_langfuse = settings.LANGFUSE_ENABLED
        
        if enable_langfuse and LANGFUSE_AVAILABLE:
            if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                self.langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST
                )
                logger.info(f"Langfuse initialized: {settings.LANGFUSE_HOST}")
            else:
                logger.warning("Langfuse enabled but keys not configured")
        
        # 初始化 Embedding
        self.embed_model = Qwen3VLEmbedding(
            api_key=self.api_key,
            model=embedding_model,
            dimension=embedding_dim
        )
        
        # 初始化 Reranker
        self.reranker = DashScopeReranker(
            api_key=self.api_key,
            model=rerank_model,
            top_n=rerank_top_n
        )
        
        # 初始化 Summarizer
        if self.enable_summary:
            self.summarizer = Qwen36Summarizer(
                api_key=self.api_key,
                model=summarizer_model
            )
        else:
            self.summarizer = None
        
        # 设置全局配置
        Settings.embed_model = self.embed_model
        
        # 初始化 Milvus (HNSW 参数修复)
        # 不使用 scalar_field_names，让 MilvusVectorStore 用默认 schema
        # 视频字段将通过直接 PyMilvus 插入来管理
        self.vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=collection_name,
            dim=embedding_dim,
            index_config={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "efConstruction": 128,  # 必须在 [8, 512]
                "M": 16
            }
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # 检查 Milvus 是否已有数据，如果有则加载
        try:
            from pymilvus import connections, Collection
            if not connections.has_connection("default"):
                connections.connect(uri=milvus_uri)
            col = Collection(collection_name)
            if col.num_entities > 0:
                # 加载已存在的索引
                self.index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    storage_context=self.storage_context
                )
                logger.info(f"Loaded existing index with {col.num_entities} documents")
            else:
                self.index = None
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self.index = None
        
        if self.index is None:
            logger.info(f"LlamaIndexMultimodalRetriever initialized: empty collection")
        else:
            logger.info(f"LlamaIndexMultimodalRetriever initialized: collection={collection_name}")
    
    def create_index(self, documents: List[Document]):
        """
        创建索引
        """
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context
        )
        # Flush to Milvus
        self._flush_to_milvus()
        logger.info(f"Created index with {len(documents)} documents")
    
    def _flush_to_milvus(self):
        """Flush data to Milvus and load collection"""
        try:
            from pymilvus import connections, Collection
            # 确保连接存在
            if not connections.has_connection("default"):
                connections.connect(uri=self.vector_store.uri)
            collection = Collection(self.vector_store.collection_name)
            collection.flush()
            collection.load()
            logger.info(f"Flushed and loaded collection: {collection.num_entities} entities")
        except Exception as e:
            logger.warning(f"Flush to Milvus failed: {e}")
    
    def _get_image_base64(self, image_url: str) -> Optional[str]:
        """将图片 URL 或本地路径转换为 base64"""
        import base64
        import os
        from pathlib import Path
        
        if not image_url:
            return None
        
        # 如果已经是 base64 data URL，直接提取
        if image_url.startswith('data:image'):
            if ',' in image_url:
                return image_url.split(',')[1]
            return image_url
        
        # 如果是本地文件路径
        if os.path.exists(image_url):
            with open(image_url, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        # 尝试作为 URL 处理 - 提取文件名
        filename = image_url.split('/')[-1]
        base_name = filename.replace('.png', '').replace('.jpg', '')
        
        # 查找本地文件
        data_dir = Path('/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval/data/xhs_images')
        for ext in ['.png', '.jpg']:
            local_path = data_dir / f"{base_name}{ext}"
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
        
        return None
    
    def add_travel_memories(
        self,
        memories: List[Dict[str, Any]]
    ):
        """
        添加旅途记忆（多模态版本）
        同时编码文本和图片/视频
        使用 PyMilvus 直接插入以支持 video_url 字段
        """
        from llama_index.core.schema import TextNode
        from pymilvus import connections, Collection
        
        # 确保 Milvus 连接
        if not connections.has_connection("default"):
            connections.connect(uri=self.vector_store.uri)
        collection = Collection(self.vector_store.collection_name)
        
        nodes = []
        success_count = 0
        
        for mem in memories:
            # 构建多模态输入（支持文本、图片、视频）
            text = mem.get("content", "")
            image_url = mem.get("image_url", "")
            video_url = mem.get("video_url", "")
            
            # 计算多模态 embedding
            try:
                import dashscope
                dashscope.api_key = self.api_key  # 确保 API key 已设置
                input_data = []
                if text:
                    input_data.append({"text": text})
                if image_url:
                    input_data.append({"image": image_url})
                if video_url:
                    input_data.append({"video": video_url})
                
                if input_data:
                    resp = dashscope.MultiModalEmbedding.call(
                        model=self.embed_model.model,
                        input=input_data,
                        dimension=self.embed_model.dimension
                    )
                    if resp.status_code == 200:
                        embedding = resp.output["embeddings"][0]["embedding"]
                    else:
                        embedding = self.embed_model._get_text_embedding(text)
                else:
                    embedding = self.embed_model._get_text_embedding(text)
                
                # 准备 Milvus 插入数据 - 使用元组格式
                tags = mem.get("tags", [])
                tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
                
                doc_id = mem.get("id", "")
                media_type = "video" if video_url else ("image" if image_url else "text")
                
                # 使用列式插入 - 每个字段一个列表
                # 注意：字段顺序必须与 schema 一致
                insert_data = [
                    [doc_id],          # doc_id
                    [text],            # text
                    [mem.get("city", "")],  # city
                    [tags_str],        # tags
                    [image_url],       # image_url
                    [video_url],       # video_url
                    [mem.get("mood", "")],  # mood
                    [mem.get("outfit", "")], # outfit
                    [media_type],      # media_type
                    [embedding]       # embedding
                ]
                
                try:
                    collection.insert(insert_data)
                    success_count += 1
                    logger.info(f"Inserted video: {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to insert {doc_id}: {e}")
                    continue
                
                # 同时创建 TextNode 用于 LlamaIndex 索引
                node = TextNode(
                    text=text,
                    embedding=embedding,
                    metadata={
                        "doc_id": doc_id,
                        "city": mem.get("city", ""),
                        "date": mem.get("date", ""),
                        "tags": mem.get("tags", []),
                        "mood": mem.get("mood", ""),
                        "outfit": mem.get("outfit", ""),
                        "source": mem.get("source", ""),
                        "image_url": image_url,
                        "video_url": video_url,
                        "media_type": media_type,
                    }
                )
                nodes.append(node)
                
            except Exception as e:
                logger.warning(f"Failed to encode memory {mem.get('id')}: {e}")
                continue
        
        if success_count == 0:
            logger.warning("No nodes to add")
            return 0
        
        # Flush 到 Milvus
        try:
            collection.flush()
            collection.load()
            logger.info(f"Inserted {success_count} documents to Milvus")
        except Exception as e:
            logger.error(f"Failed to flush to Milvus: {e}")
            return 0
        
        # 同时更新 LlamaIndex 索引
        if self.index is None:
            from llama_index.core import VectorStoreIndex
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
            )
        else:
            for node in nodes:
                self.index.insert_nodes([node])
        
        return success_count
    
    def add_documents(self, documents: List[Document]):
        """添加文档 (兼容接口)"""
        if self.index is None:
            self.create_index(documents)
        else:
            for doc in documents:
                self.index.insert(doc)
            self._flush_to_milvus()
    
    # ========== 兼容自定义引擎的 API 接口 ==========
    
    async def search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[str] = None,
        query_images: Optional[List[str]] = None,
        query_video: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        top_k: int = 10,
        use_rerank: bool = True,
        trace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        统一检索入口 (兼容自定义引擎接口)
        集成 Langfuse 追踪
        """
        import time
        start_time = time.time()
        
        if self.index is None:
            raise ValueError("Index not created. Please add documents first.")
        
        # 构建查询
        query = query_text or ""
        
        # 创建检索器 (粗筛更多结果用于 rerank)
        retrieval_k = top_k * 3 if use_rerank else top_k
        
        # =============================================
        # 处理视频查询 - Qwen3-VL 原生支持
        # =============================================
        if query_video:
            try:
                import dashscope
                resp = dashscope.MultiModalEmbedding.call(
                    model=self.embed_model.model,
                    input=[{"video": query_video}],
                    dimension=self.embed_model.dimension
                )
                if resp.status_code == 200:
                    query_emb = resp.output["embeddings"][0]["embedding"]
                    # 使用视频向量搜索
                    from pymilvus import connections, Collection
                    if not connections.has_connection("default"):
                        connections.connect(host="localhost", port="19530")
                    collection = Collection("duckyo_travel_memories")
                    collection.load()
                    
                    search_params = {"metric_type": "COSINE", "params": {}}
                    milvus_results = collection.search(
                        data=[query_emb],
                        anns_field="embedding",
                        param=search_params,
                        limit=retrieval_k,
                        output_fields=["doc_id", "text", "image_url", "video_url", "city", "tags"]
                    )
                    
                    # 格式化结果
                    results = []
                    for hits in milvus_results:
                        for hit in hits:
                            results.append({
                                "doc_id": hit.entity.get("doc_id"),
                                "text": hit.entity.get("text"),
                                "image_url": hit.entity.get("image_url"),
                                "video_url": hit.entity.get("video_url"),
                                "city": hit.entity.get("city"),
                                "tags": hit.entity.get("tags"),
                                "vector_score": hit.distance,
                            })
                    return results[:top_k]
            except Exception as e:
                logger.warning(f"Video query failed: {e}")
        

        # =============================================
        # 处理图片查询 - 使用本地 base64
        # =============================================
        image_nodes = None
        if query_image:
            image_base64 = self._get_image_base64(query_image)
            if image_base64:
                try:
                    # 使用图片+文本混合 embedding 搜索（与入库时保持一致）
                    # 图搜图时，query 文本为空，只使用图片
                    query_emb = self.embed_model._get_multimodal_embedding(
                        text="",
                        image=f"data:image/png;base64,{image_base64}"
                    )
                    
                    # 直接使用 Milvus 搜索
                    from pymilvus import connections, Collection
                    if not connections.has_connection("default"):
                        connections.connect(host="localhost", port="19530")
                    collection = Collection("duckyo_travel_memories")
                    collection.load()
                    
                    search_params = {"metric_type": "COSINE", "params": {}}
                    milvus_results = collection.search(
                        data=[query_emb],
                        anns_field="embedding",
                        param=search_params,
                        limit=retrieval_k,
                        output_fields=["doc_id", "text", "image_url", "video_url"]
                    )
                    
                    # 重建 nodes（包含 metadata）
                    from llama_index.core.schema import TextNode
                    image_nodes = []
                    for hits in milvus_results:
                        for hit in hits:
                            text = hit.entity.get("text", "")
                            doc_id = hit.entity.get("doc_id")
                            
                            if text:
                                # 查询 image_url
                                doc_results = collection.query(
                                    expr=f'doc_id == "{doc_id}"',
                                    output_fields=["doc_id", "text", "image_url", "video_url"],
                                    limit=1
                                )
                                image_url = ""
                                if doc_results:
                                    image_url = doc_results[0].get("image_url", "")
                                
                                # 创建带 metadata 的 TextNode，并包装成 NodeWithScore
                                from llama_index.core.schema import NodeWithScore
                                text_node = TextNode(
                                    text=text, 
                                    embedding=query_emb,
                                    metadata={"image_url": image_url}
                                )
                                # 使用 Milvus 返回的距离作为 score
                                score = hit.distance if hasattr(hit, 'distance') else 0.5
                                node_with_score = NodeWithScore(node=text_node, score=score)
                                image_nodes.append(node_with_score)
                    
                    logger.info(f"Image search: retrieved {len(image_nodes)} nodes")
                except Exception as e:
                    logger.warning(f"Image search failed: {e}")
                    image_nodes = None
        
        # 如果有图片搜索结果，直接使用；否则用文本搜索
        image_search_used = False
        if image_nodes is not None and len(image_nodes) > 0:
            nodes = image_nodes
            image_search_used = True
        else:
            # 只有当有文本查询时才执行文本搜索
            if query and query.strip():
                retriever = VectorIndexRetriever(index=self.index, similarity_top_k=retrieval_k)
                nodes = retriever.retrieve(query)
            else:
                # 无文本查询，使用空的 nodes 列表
                nodes = []

        # 如果使用了图片搜索，跳过后续的文本检索逻辑
        lf_trace = None
        lf_span = None
        if image_search_used:
            vector_count = len(nodes)
        else:
            # Langfuse: 开始追踪
            if self.langfuse:
                lf_trace = self.langfuse.trace(
                    name="multimodal_search",
                    id=trace_id,
                    input={"query": query, "top_k": top_k, "use_rerank": use_rerank}
                )
                lf_span = lf_trace.span(name="vector_retrieval")
            
            vector_count = len(nodes)
            
            # Langfuse: 记录向量检索结果
            if lf_span:
                lf_span.end(output={"retrieved_count": vector_count})
                lf_span = lf_trace.span(name="rerank" if use_rerank else "format")
        
        # Langfuse: 记录向量检索结果
        if lf_span:
            lf_span.end(output={"retrieved_count": vector_count})
            lf_span = lf_trace.span(name="rerank" if use_rerank else "format")
        
        # Rerank (暂时禁用，因为 API 对表情符号支持不好)
        # if use_rerank and len(nodes) > 1:
        if False:  # 禁用 Rerank
            from llama_index.core.schema import QueryBundle
            
            # 如果是图搜图，设置 query_image 给 Rerank
            if image_search_used and query_image:
                image_base64 = self._get_image_base64(query_image)
                if image_base64:
                    self.reranker._query_image = f"data:image/png;base64,{image_base64}"
            
            query_bundle = QueryBundle(query_str=query)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        
        # 获取节点的 doc_id 和视频信息 - 通过文本内容匹配
        # 因为直接用 PyMilvus 插入时 metadata 为空
        doc_id_map = {}
        video_info_map = {}  # 存储 video_url 和 media_type
        try:
            from pymilvus import connections, Collection
            collection = Collection('duckyo_travel_memories')
            if collection.num_entities > 0:
                # 收集所有唯一文本
                unique_texts = list(set([node.node.text[:200] for node in nodes]))
                # 直接从 Milvus 查询所有记录，构建 doc_id 到 city 的映射
                try:
                    all_results = collection.query(
                        expr='',
                        output_fields=['doc_id', 'text', 'city', 'image_url', 'video_url', 'media_type'],
                        limit=100
                    )
                    for r in all_results:
                        text_key = r['text'][:200]
                        video_info_map[text_key] = {
                            'doc_id': r['doc_id'],
                            'city': r.get('city', ''),
                            'video_url': r.get('video_url', ''),
                            'media_type': r.get('media_type', 'text'),
                            'image_url': r.get('image_url', '')
                        }
                except Exception as e:
                    logger.warning(f"Failed to query Milvus: {e}")
        except Exception as e:
            logger.warning(f"Failed to get doc_id mapping: {e}")

        # 格式化结果 (兼容原有格式)
        results = []
        for node in nodes:
            # 处理 TextNode 和 NodeWithScore
            if hasattr(node, 'node'):
                node_obj = node.node
                node_id = node_obj.id_
                node_text = node_obj.text
                meta = node_obj.metadata if hasattr(node_obj, 'metadata') else {}
            else:
                node_id = node.id_
                node_text = node.text
                meta = node.metadata if hasattr(node, 'metadata') else {}
            
            # 优先使用 Milvus 查询到的信息
            text_key = node_text[:200]
            milvus_info = video_info_map.get(text_key, {})
            
            result_id = milvus_info.get('doc_id') or doc_id_map.get(node_id, node_id)
            city = milvus_info.get('city') or meta.get('city', '')
            video_url = milvus_info.get('video_url') or meta.get('video_url', '')
            media_type = milvus_info.get('media_type') or meta.get('media_type', 'text')
            image_url = milvus_info.get('image_url') or meta.get('image_url', '')
            
            results.append({
                "id": result_id,
                "content": node_text,
                "score": getattr(node, 'score', 1.0),
                "vector_score": getattr(node, 'score', 1.0),
                "rerank_score": getattr(node, 'score', 1.0) if use_rerank else None,
                "file_url": image_url,
                "image_url": image_url,
                "video_url": video_url,
                "city": city,
                "country": meta.get("country", ""),
                "date": meta.get("date", ""),
                "tags": meta.get("tags", []),
                "mood": meta.get("mood", ""),
                "outfit": meta.get("outfit", ""),
                "media_type": media_type,
            })
        
        # Langfuse: 记录最终结果
        latency_ms = (time.time() - start_time) * 1000
        if lf_trace:
            if lf_span:
                lf_span.end(output={"final_count": len(results)})
            
            # 添加评分指标
            lf_trace.score(name="result_count", value=len(results))
            lf_trace.score(name="latency_ms", value=latency_ms)
            lf_trace.score(name="vector_retrieved", value=vector_count)
            
            lf_trace.end(output={"results": results[:3]})  # 只记录前3条
        
        return results
    
    async def search_with_summary(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        搜索并生成总结
        
        Returns:
            {
                "results": [...],
                "summary": "...",
                "total": int
            }
        """
        # 执行搜索
        results = await self.search(
            query_text=query,
            top_k=top_k,
            use_rerank=use_rerank
        )
        
        # 生成总结
        summary = None
        if self.summarizer and results:
            summary = await self.summarizer.asummarize_results(query, results)
        
        return {
            "results": results,
            "summary": summary,
            "total": len(results)
        }
    
    async def text_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """文搜图"""
        return await self.search(query_text=query, filters=filters, top_k=top_k or 10)
    
    async def image_search(
        self,
        image_url: str,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """图搜图 - 使用图片 embedding 进行检索"""
        # 使用图片作为查询，通过 query_image 参数传递
        return await self.search(query_text="", query_image=image_url, filters=filters, top_k=top_k or 10)
    
    async def multi_image_search(
        self,
        image_urls: List[str],
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """多图搜图"""
        return await self.search(query_text=f"images:{','.join(image_urls)}", filters=filters, top_k=top_k or 10)
    
    async def hybrid_search(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """图文混合搜索"""
        query = text or ""
        if image:
            query += f" image:{image}"
        if images:
            query += f" images:{','.join(images)}"
        return await self.search(query_text=query, filters=filters, top_k=top_k or 10)
    
    async def ingest(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 50
    ):
        """
        批量入库 (兼容自定义引擎接口)
        """
        self.add_travel_memories(items)
        logger.info(f"Ingested {len(items)} items via LlamaIndex")
    
    # ========== 原有方法 ==========
    
    def search_sync(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        同步搜索 (原有方法)
        """
        if self.index is None:
            raise ValueError("Index not created. Please add documents first.")
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )
        
        nodes = retriever.retrieve(query)
        
        results = []
        for node in nodes:
            results.append({
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
                "id": node.node.id_
            })
        
        return results

# 全局实例
_llamaindex_retriever: Optional[LlamaIndexMultimodalRetriever] = None

def get_llamaindex_retriever() -> LlamaIndexMultimodalRetriever:
    """获取全局 LlamaIndex 检索器实例"""
    global _llamaindex_retriever
    if _llamaindex_retriever is None:
        from config.settings import settings
        _llamaindex_retriever = LlamaIndexMultimodalRetriever(
            api_key=settings.DASHSCOPE_API_KEY,
            collection_name=settings.MILVUS_COLLECTION
        )
    return _llamaindex_retriever

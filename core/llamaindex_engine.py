"""
LlamaIndex multimodal retrieval engine
Using Qwen3-VL Embedding + Milvus + Langfuse observability
"""
import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import dashscope
from loguru import logger

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import ImageDocument, NodeWithScore, QueryBundle, TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import Collection, connections

from config.settings import get_settings

# Optional Langfuse integration
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIM = 2560
DEFAULT_EMBEDDING_MODEL = "qwen3-vl-embedding"
DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_SUMMARIZER_MODEL = "qwen3.6-plus"
MAX_RESULTS_FOR_SUMMARY = 10
MAX_CONTENT_PREVIEW = 200
DEFAULT_SUMMARY_LENGTH = 500
RERANK_COARSE_MULTIPLIER = 3
MAX_TEXT_LENGTH_FOR_API = 500
HNSW_EF_CONSTRUCTION = 128
HNSW_M = 16

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetrievalError(Exception):
    """Base retrieval engine error."""


class EmbeddingError(RetrievalError):
    """Embedding API call failed."""


class RerankError(RetrievalError):
    """Rerank API call failed."""


class MilvusError(RetrievalError):
    """Milvus operation failed."""


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class Qwen36Summarizer:
    """Search result summarizer using qwen3.6-plus."""

    def __init__(self, api_key: str, model: str = DEFAULT_SUMMARIZER_MODEL):
        self.api_key = api_key
        self.model = model
        logger.info(f"Qwen36Summarizer initialized: model={model}")

    def summarize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_length: int = DEFAULT_SUMMARY_LENGTH,
    ) -> str:
        if not results:
            return "No relevant results found."

        results_text = []
        for i, r in enumerate(results[:MAX_RESULTS_FOR_SUMMARY], 1):
            content = r.get("content", "")[:MAX_CONTENT_PREVIEW]
            city = r.get("city", "Unknown location")
            score = r.get("score", 0)
            results_text.append(
                f"[{i}] {city}: {content}... (Relevance: {score:.2f})"
            )

        results_str = "\n".join(results_text)
        prompt = (
            f"Based on the following search results, generate a concise summary for the user query.\n\n"
            f"User query: {query}\n\n"
            f"Search results:\n{results_str}\n\n"
            f"Please generate a concise summary including:\n"
            f"1. Main content found\n"
            f"2. Locations involved\n"
            f"3. Overall atmosphere or theme\n\n"
            f"Summary:"
        )

        try:
            resp = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "text": "You are an intelligent assistant, skilled at summarizing search results."
                            }
                        ],
                    },
                    {"role": "user", "content": [{"text": prompt}]},
                ],
                api_key=self.api_key,
            )

            if resp.status_code == 200:
                content = resp.output.choices[0].message.content
                if isinstance(content, list) and len(content) > 0:
                    summary = content[0].get("text", "")
                else:
                    summary = str(content)
                return summary[:max_length]
            else:
                logger.warning(f"Summarization failed: {resp.message}")
                return self._fallback_summary(results)
        except dashscope.errors.ApiException as e:
            logger.warning(f"Summarization API error: {e}")
            return self._fallback_summary(results)
        except ConnectionError as e:
            logger.warning(f"Summarization connection error: {e}")
            return self._fallback_summary(results)

    def _fallback_summary(self, results: List[Dict[str, Any]]) -> str:
        cities = {r.get("city") for r in results if r.get("city")}
        city_str = ", ".join(cities) if cities else "Multiple locations"
        return f"Found {len(results)} relevant memories, involving locations: {city_str}."

    async def asummarize_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_length: int = DEFAULT_SUMMARY_LENGTH,
    ) -> str:
        """Async wrapper around the synchronous summarizer."""
        return await asyncio.to_thread(
            self.summarize_results, query, results, max_length
        )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class Qwen3VLEmbedding(BaseEmbedding):
    """LlamaIndex compatible Qwen3-VL Embedding."""

    model: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIM

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        **kwargs: Any,
    ):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "dimension", dimension)
        self._api_key = api_key
        super().__init__(**kwargs)
        logger.info(f"Qwen3VLEmbedding initialized: model={model}, dim={dimension}")

    def _call_embedding(self, input_list: List[Dict[str, Any]]) -> List[float]:
        """Internal helper to call the embedding API."""
        resp = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=input_list,
            dimension=self.dimension,
            api_key=self._api_key,
        )
        if resp.status_code != 200:
            raise EmbeddingError(f"Embedding failed: {resp.message}")
        return resp.output["embeddings"][0]["embedding"]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._call_embedding([{"text": text}])

    def _get_image_embedding(self, image_url: str) -> List[float]:
        return self._call_embedding([{"image": image_url}])

    def _get_multimodal_embedding(self, text: str, image: str) -> List[float]:
        input_list: List[Dict[str, Any]] = []
        if text:
            input_list.append({"text": text})
        if image:
            input_list.append({"image": image})
        if not input_list:
            raise EmbeddingError("At least one of text or image must be provided")
        return self._call_embedding(input_list)

    def _get_image_base64(self, image_url: str) -> Optional[str]:
        """Convert image URL or local path to base64."""
        if not image_url:
            return None

        if image_url.startswith("data:image"):
            if "," in image_url:
                return image_url.split(",", 1)[1]
            return image_url

        if os.path.exists(image_url):
            with open(image_url, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        filename = Path(urlparse(image_url).path).name
        if not filename:
            return None

        base_name = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
        data_dir = Path(get_settings().DATA_DIR) / "xhs_images"
        for ext in (".png", ".jpg", ".jpeg"):
            local_path = data_dir / f"{base_name}{ext}"
            if local_path.exists():
                with open(local_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

        return None

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    async def _aget_image_embedding(self, image_url: str) -> List[float]:
        return await asyncio.to_thread(self._get_image_embedding, image_url)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class DashScopeReranker(BaseNodePostprocessor):
    """LlamaIndex compatible DashScope Reranker."""

    model: str = DEFAULT_RERANK_MODEL
    top_n: int = 10
    _api_key: str = ""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        top_n: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._api_key = api_key
        self.model = model
        self.top_n = top_n
        self._query_image: Optional[str] = None
        logger.info(f"DashScopeReranker initialized: model={model}, top_n={top_n}")

    @staticmethod
    def _clean_text_for_api(text: str) -> str:
        if not text:
            return "query"
        # Strip emoji by filtering to ASCII + common multilingual ranges
        cleaned = "".join(
            ch
            for ch in text
            if ord(ch) < 0x2600 or 0xE000 < ord(ch) < 0xF900
        ).strip()
        return (cleaned or text)[:MAX_TEXT_LENGTH_FOR_API]

    def _call_rerank_api(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_image: Optional[str] = None,
    ) -> Optional[List[Dict]]:
        try:
            clean_query = self._clean_text_for_api(query)
            query_input = (
                {"image": query_image} if query_image else {"text": clean_query}
            )

            resp = dashscope.TextReRank.call(
                model=self.model,
                query=query_input,
                documents=documents,
                top_n=min(self.top_n, len(documents)),
                api_key=self._api_key,
            )

            if resp.status_code == 200:
                return resp.output.get("results", [])
            else:
                logger.warning(f"Rerank API failed: {resp.message}")
                return None
        except dashscope.errors.ApiException as e:
            logger.warning(f"Rerank API error: {e}")
            return None
        except ConnectionError as e:
            logger.warning(f"Rerank connection error: {e}")
            return None

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        query = query_bundle.query_str
        documents: List[Dict[str, Any]] = []

        for node in nodes:
            node_obj = node.node if hasattr(node, "node") else node
            text = node_obj.text if hasattr(node_obj, "text") else str(node_obj)
            metadata = node_obj.metadata if hasattr(node_obj, "metadata") else {}
            image_url = metadata.get("image_url", "")

            if image_url and image_url.startswith("http"):
                try:
                    parsed = urlparse(image_url)
                    hostname = parsed.hostname or ""
                    if not (
                        hostname in ("localhost", "127.0.0.1")
                        or hostname.startswith("192.168.")
                        or hostname.startswith("10.")
                    ):
                        documents.append({"image": image_url})
                        continue
                except ValueError:
                    pass
            documents.append({"text": self._clean_text_for_api(text)})

        try:
            results = self._call_rerank_api(query, documents, self._query_image)
            self._query_image = None

            if not results:
                logger.warning("Rerank API returned no results, falling back to vector scores")
                return nodes[: self.top_n]

            reranked = []
            for r in results:
                idx = r.get("index", 0)
                if 0 <= idx < len(nodes):
                    nodes[idx].score = r.get("relevance_score", nodes[idx].score)
                    reranked.append(nodes[idx])

            logger.info(f"Reranked {len(nodes)} nodes to {len(reranked)} using {self.model}")
            return reranked
        except RetrievalError:
            return nodes[: self.top_n]


Qwen3VLReranker = DashScopeReranker


# ---------------------------------------------------------------------------
# Milvus connection context manager
# ---------------------------------------------------------------------------


class _MilvusConnection:
    """Context manager for temporary Milvus connections."""

    def __init__(self, uri: str):
        self.uri = uri
        self._alias = f"conn_{id(self)}"

    def __enter__(self) -> "_MilvusConnection":
        connections.connect(alias=self._alias, uri=self.uri)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            connections.disconnect(alias=self._alias)
        except Exception:
            pass

    def collection(self, name: str) -> Collection:
        return Collection(name, using=self._alias)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class LlamaIndexMultimodalRetriever:
    """LlamaIndex multimodal retrieval engine."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        milvus_uri: str = "http://localhost:19530",
        collection_name: str = "duckyo_travel_memories",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_top_n: int = 10,
        summarizer_model: str = DEFAULT_SUMMARIZER_MODEL,
        enable_summary: bool = True,
        enable_langfuse: Optional[bool] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.DASHSCOPE_API_KEY or ""
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.rerank_top_n = rerank_top_n
        self.enable_summary = enable_summary

        # Langfuse
        self.langfuse = None
        if enable_langfuse is None:
            enable_langfuse = settings.LANGFUSE_ENABLED
        if enable_langfuse and LANGFUSE_AVAILABLE:
            if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                self.langfuse = Langfuse(
                    public_key=settings.LANGFUSE_PUBLIC_KEY,
                    secret_key=settings.LANGFUSE_SECRET_KEY,
                    host=settings.LANGFUSE_HOST,
                )
                logger.info(f"Langfuse initialized: {settings.LANGFUSE_HOST}")
            else:
                logger.warning("Langfuse enabled but keys not configured")

        # Embedding
        self.embed_model = Qwen3VLEmbedding(
            api_key=self.api_key,
            model=embedding_model,
            dimension=embedding_dim,
        )
        Settings.embed_model = self.embed_model

        # Reranker
        self.reranker = DashScopeReranker(
            api_key=self.api_key,
            model=rerank_model,
            top_n=rerank_top_n,
        )

        # Summarizer
        self.summarizer = (
            Qwen36Summarizer(api_key=self.api_key, model=summarizer_model)
            if self.enable_summary
            else None
        )

        # Vector store
        self.vector_store = MilvusVectorStore(
            uri=milvus_uri,
            collection_name=collection_name,
            dim=embedding_dim,
            index_config={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "efConstruction": HNSW_EF_CONSTRUCTION,
                "M": HNSW_M,
            },
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # Load existing index if collection has data
        self.index = self._load_existing_index()

    def _load_existing_index(self) -> Optional[VectorStoreIndex]:
        try:
            with _MilvusConnection(self.milvus_uri) as conn:
                col = conn.collection(self.collection_name)
                if col.num_entities > 0:
                    index = VectorStoreIndex.from_vector_store(
                        self.vector_store,
                        storage_context=self.storage_context,
                    )
                    logger.info(f"Loaded existing index with {col.num_entities} documents")
                    return index
        except MilvusError as e:
            logger.warning(f"Failed to load existing index: {e}")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
        logger.info("LlamaIndexMultimodalRetriever initialized: empty collection")
        return None

    def create_index(self, documents: List[Document]) -> None:
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )
        self._flush_to_milvus()
        logger.info(f"Created index with {len(documents)} documents")

    def _flush_to_milvus(self) -> None:
        try:
            with _MilvusConnection(self.milvus_uri) as conn:
                collection = conn.collection(self.vector_store.collection_name)
                collection.flush()
                collection.load()
                logger.info(
                    f"Flushed and loaded collection: {collection.num_entities} entities"
                )
        except Exception as e:
            logger.warning(f"Flush to Milvus failed: {e}")

    def _get_image_base64(self, image_url: str) -> Optional[str]:
        return self.embed_model._get_image_base64(image_url)

    def add_travel_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Add travel memories (multimodal version)."""
        nodes: List[TextNode] = []
        success_count = 0

        with _MilvusConnection(self.milvus_uri) as conn:
            collection = conn.collection(self.collection_name)

            for mem in memories:
                text = mem.get("content", "")
                image_url = mem.get("image_url", "")
                video_url = mem.get("video_url", "")

                try:
                    embedding = self._encode_memory(text, image_url, video_url)
                except EmbeddingError as e:
                    logger.warning(f"Failed to encode memory {mem.get('id')}: {e}")
                    continue

                tags = mem.get("tags", [])
                tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
                doc_id = mem.get("id", "")
                media_type = "video" if video_url else ("image" if image_url else "text")

                insert_data = [
                    [doc_id],
                    [text],
                    [mem.get("city", "")],
                    [tags_str],
                    [image_url],
                    [video_url],
                    [mem.get("mood", "")],
                    [mem.get("outfit", "")],
                    [media_type],
                    [embedding],
                ]

                try:
                    collection.insert(insert_data)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to insert {doc_id}: {e}")
                    continue

                meta = {
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
                nodes.append(TextNode(text=text, embedding=embedding, metadata=meta))

            if success_count:
                try:
                    collection.flush()
                    collection.load()
                    logger.info(f"Inserted {success_count} documents to Milvus")
                except Exception as e:
                    logger.error(f"Failed to flush to Milvus: {e}")
                    return 0

        if self.index is None and nodes:
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=self.storage_context,
            )
        elif nodes:
            for node in nodes:
                self.index.insert_nodes([node])  # type: ignore[union-attr]

        return success_count

    def _encode_memory(
        self, text: str, image_url: str, video_url: str
    ) -> List[float]:
        input_data: List[Dict[str, Any]] = []
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
                dimension=self.embed_model.dimension,
                api_key=self.api_key,
            )
            if resp.status_code == 200:
                return resp.output["embeddings"][0]["embedding"]
            # Fallback to text-only embedding
            if text:
                return self.embed_model._get_text_embedding(text)
            raise EmbeddingError(f"Embedding failed: {resp.message}")
        if text:
            return self.embed_model._get_text_embedding(text)
        raise EmbeddingError("No content provided for embedding")

    def add_documents(self, documents: List[Document]) -> None:
        if self.index is None:
            self.create_index(documents)
        else:
            for doc in documents:
                self.index.insert(doc)  # type: ignore[union-attr]
            self._flush_to_milvus()

    # -----------------------------------------------------------------------
    # Search methods
    # -----------------------------------------------------------------------

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
        trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Unified search entry (compatible with custom engine interface)."""
        start_time = time.time()
        if self.index is None:
            raise ValueError("Index not created. Please add documents first.")

        retrieval_k = top_k * RERANK_COARSE_MULTIPLIER if use_rerank else top_k
        query = query_text or ""

        # Langfuse trace setup
        lf_trace = None
        lf_span = None
        if self.langfuse:
            lf_trace = self.langfuse.trace(
                name="multimodal_search",
                id=trace_id,
                input={
                    "query": query,
                    "top_k": top_k,
                    "use_rerank": use_rerank,
                },
            )

        nodes: List[NodeWithScore] = []
        image_search_used = False

        try:
            if query_video:
                nodes = await self._search_video(query_video, retrieval_k, lf_trace)
                if nodes:
                    return self._format_results(nodes, top_k)

            if query_image:
                img_nodes = await self._search_image(query_image, retrieval_k)
                if img_nodes:
                    nodes = img_nodes
                    image_search_used = True

            if not image_search_used and query.strip():
                nodes = await self._search_text(query, retrieval_k, lf_trace)

            # Rerank (currently disabled by configuration)
            if use_rerank and len(nodes) > 1:
                nodes = await self._rerank_nodes(query, query_image, nodes, lf_trace)

            results = self._format_results(nodes, top_k)

            latency_ms = (time.time() - start_time) * 1000
            if lf_trace:
                lf_trace.score(name="result_count", value=len(results))
                lf_trace.score(name="latency_ms", value=latency_ms)
                lf_trace.score(name="vector_retrieved", value=len(nodes))
                lf_trace.end(output={"results": results[:3]})

            return results
        except Exception:
            if lf_trace:
                lf_trace.end()
            raise

    async def _search_video(
        self, video_url: str, retrieval_k: int, lf_trace: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Video search: encode video and query Milvus directly."""
        try:
            emb = await asyncio.to_thread(
                self._encode_memory, "", "", video_url
            )
            with _MilvusConnection(self.milvus_uri) as conn:
                collection = conn.collection(self.collection_name)
                collection.load()
                search_params = {"metric_type": "COSINE", "params": {}}
                milvus_results = collection.search(
                    data=[emb],
                    anns_field="embedding",
                    param=search_params,
                    limit=retrieval_k,
                    output_fields=[
                        "doc_id",
                        "text",
                        "image_url",
                        "video_url",
                        "city",
                        "tags",
                    ],
                )
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
                return results
        except Exception as e:
            logger.warning(f"Video query failed: {e}")
        return []

    async def _search_image(
        self, image_url: str, retrieval_k: int
    ) -> List[NodeWithScore]:
        """Image search: encode image and query Milvus directly."""
        image_base64 = self._get_image_base64(image_url)
        if not image_base64:
            return []
        try:
            query_emb = await asyncio.to_thread(
                self.embed_model._get_multimodal_embedding,
                "",
                f"data:image/png;base64,{image_base64}",
            )
            with _MilvusConnection(self.milvus_uri) as conn:
                collection = conn.collection(self.collection_name)
                collection.load()
                search_params = {"metric_type": "COSINE", "params": {}}
                milvus_results = collection.search(
                    data=[query_emb],
                    anns_field="embedding",
                    param=search_params,
                    limit=retrieval_k,
                    output_fields=["doc_id", "text", "image_url", "video_url"],
                )
                nodes: List[NodeWithScore] = []
                for hits in milvus_results:
                    for hit in hits:
                        text = hit.entity.get("text", "")
                        doc_id = hit.entity.get("doc_id")
                        image_url_val = ""
                        if doc_id:
                            try:
                                doc_results = collection.query(
                                    expr=f'doc_id == "{doc_id}"',
                                    output_fields=["image_url"],
                                    limit=1,
                                )
                                if doc_results:
                                    image_url_val = doc_results[0].get("image_url", "")
                            except Exception:
                                pass
                        text_node = TextNode(
                            text=text,
                            embedding=query_emb,
                            metadata={"image_url": image_url_val},
                        )
                        nodes.append(NodeWithScore(node=text_node, score=hit.distance))
                logger.info(f"Image search: retrieved {len(nodes)} nodes")
                return nodes
        except Exception as e:
            logger.warning(f"Image search failed: {e}")
        return []

    async def _search_text(
        self, query: str, retrieval_k: int, lf_trace: Optional[Any]
    ) -> List[NodeWithScore]:
        """Text search via LlamaIndex retriever."""
        if lf_trace:
            span = lf_trace.span(name="vector_retrieval")
        else:
            span = None
        try:
            retriever = VectorIndexRetriever(
                index=self.index,  # type: ignore[arg-type]
                similarity_top_k=retrieval_k,
            )
            nodes = retriever.retrieve(query)
            if span:
                span.end(output={"retrieved_count": len(nodes)})
            return nodes
        except Exception:
            if span:
                span.end()
            raise

    async def _rerank_nodes(
        self,
        query: str,
        query_image: Optional[str],
        nodes: List[NodeWithScore],
        lf_trace: Optional[Any],
    ) -> List[NodeWithScore]:
        """Rerank nodes using the configured reranker."""
        if lf_trace:
            span = lf_trace.span(name="rerank")
        else:
            span = None
        try:
            if query_image:
                image_base64 = self._get_image_base64(query_image)
                if image_base64:
                    self.reranker._query_image = f"data:image/png;base64,{image_base64}"

            query_bundle = QueryBundle(query_str=query)
            nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
            if span:
                span.end(output={"reranked_count": len(nodes)})
            return nodes
        except Exception as e:
            logger.warning(f"Rerank error: {e}")
            if span:
                span.end()
            return nodes[: self.rerank_top_n]

    def _format_results(
        self, nodes: List[Any], top_k: int
    ) -> List[Dict[str, Any]]:
        """Format nodes into the canonical result dict."""
        # Build a lightweight lookup from Milvus for metadata enrichment
        video_info_map: Dict[str, Dict[str, Any]] = {}
        if nodes:
            try:
                with _MilvusConnection(self.milvus_uri) as conn:
                    collection = conn.collection(self.collection_name)
                    if collection.num_entities > 0:
                        all_results = collection.query(
                            expr="",
                            output_fields=[
                                "doc_id",
                                "text",
                                "city",
                                "image_url",
                                "video_url",
                                "media_type",
                            ],
                            limit=100,
                        )
                        for r in all_results:
                            video_info_map[r["text"][:200]] = {
                                "doc_id": r["doc_id"],
                                "city": r.get("city", ""),
                                "video_url": r.get("video_url", ""),
                                "media_type": r.get("media_type", "text"),
                                "image_url": r.get("image_url", ""),
                            }
            except Exception as e:
                logger.warning(f"Failed to query Milvus for metadata: {e}")

        results = []
        for node in nodes:
            if hasattr(node, "node"):
                node_obj = node.node
                meta = node_obj.metadata if hasattr(node_obj, "metadata") else {}
                node_text = node_obj.text
            elif isinstance(node, dict):
                # Already formatted (from video search)
                results.append(node)
                continue
            else:
                node_obj = node
                meta = node_obj.metadata if hasattr(node_obj, "metadata") else {}
                node_text = node_obj.text

            text_key = node_text[:200]
            milvus_info = video_info_map.get(text_key, {})

            result_id = milvus_info.get("doc_id") or meta.get("doc_id", "")
            city = milvus_info.get("city") or meta.get("city", "")
            video_url = milvus_info.get("video_url") or meta.get("video_url", "")
            media_type = milvus_info.get("media_type") or meta.get("media_type", "text")
            image_url = milvus_info.get("image_url") or meta.get("image_url", "")

            score = getattr(node, "score", 1.0)
            results.append({
                "id": result_id,
                "content": node_text,
                "score": score,
                "vector_score": score,
                "rerank_score": None,
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

        return results[:top_k]

    # -----------------------------------------------------------------------
    # Convenience wrappers
    # -----------------------------------------------------------------------

    async def search_with_summary(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> Dict[str, Any]:
        results = await self.search(
            query_text=query, top_k=top_k, use_rerank=use_rerank
        )
        summary = None
        if self.summarizer and results:
            summary = await self.summarizer.asummarize_results(query, results)
        return {"results": results, "summary": summary, "total": len(results)}

    async def text_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        return await self.search(
            query_text=query, filters=filters, top_k=top_k or 10
        )

    async def image_search(
        self,
        image_url: str,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        return await self.search(
            query_text="", query_image=image_url, filters=filters, top_k=top_k or 10
        )

    async def multi_image_search(
        self,
        image_urls: List[str],
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        return await self.search(
            query_text=f"images:{','.join(image_urls)}",
            filters=filters,
            top_k=top_k or 10,
        )

    async def hybrid_search(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        query = text or ""
        if image:
            query += f" image:{image}"
        if images:
            query += f" images:{','.join(images)}"
        return await self.search(query_text=query, filters=filters, top_k=top_k or 10)

    async def ingest(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 50,
    ) -> None:
        self.add_travel_memories(items)
        logger.info(f"Ingested {len(items)} items via LlamaIndex")

    def search_sync(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Synchronous search (original method)."""
        if self.index is None:
            raise ValueError("Index not created. Please add documents first.")
        retriever = VectorIndexRetriever(
            index=self.index,  # type: ignore[arg-type]
            similarity_top_k=top_k,
        )
        nodes = retriever.retrieve(query)
        return [
            {
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
                "id": node.node.id_,
            }
            for node in nodes
        ]


# ---------------------------------------------------------------------------
# Global singleton factory
# ---------------------------------------------------------------------------

_llamaindex_retriever: Optional[LlamaIndexMultimodalRetriever] = None


def get_llamaindex_retriever() -> LlamaIndexMultimodalRetriever:
    global _llamaindex_retriever
    if _llamaindex_retriever is None:
        settings = get_settings()
        _llamaindex_retriever = LlamaIndexMultimodalRetriever(
            api_key=settings.DASHSCOPE_API_KEY or "",
            collection_name=settings.MILVUS_COLLECTION,
        )
    return _llamaindex_retriever


__all__ = [
    "Qwen3VLEmbedding",
    "DashScopeReranker",
    "Qwen3VLReranker",
    "Qwen36Summarizer",
    "LlamaIndexMultimodalRetriever",
    "get_llamaindex_retriever",
    "RetrievalError",
    "EmbeddingError",
    "RerankError",
    "MilvusError",
]

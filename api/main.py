"""
FastAPI service for multimodal retrieval
"""
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core.llamaindex_engine import LlamaIndexMultimodalRetriever, get_llamaindex_retriever


class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    query_video: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=10, ge=1, le=100)
    use_rerank: bool = True


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    query: str


class IngestRequest(BaseModel):
    items: List[Dict[str, Any]]


retriever: Optional[LlamaIndexMultimodalRetriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")
    retriever = LlamaIndexMultimodalRetriever(api_key=api_key)
    yield


app = FastAPI(
    title="Multimodal Retrieval Engine",
    description="Hybrid vector search with Qwen3-VL + Milvus",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        results = await retriever.search(
            query_text=req.query_text,
            query_image=req.query_image,
            query_video=req.query_video,
            filters=req.filters,
            top_k=req.top_k,
            use_rerank=req.use_rerank,
        )
        return SearchResponse(
            results=results,
            total=len(results),
            query=req.query_text or req.query_image or req.query_video or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text")
async def search_text(
    q: str = Query(..., description="Search text"),
    city: Optional[str] = None,
    top_k: int = Query(default=10, ge=1, le=100),
) -> SearchResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    filters = {"city": city} if city else None
    results = await retriever.text_search(query=q, filters=filters, top_k=top_k)
    return SearchResponse(results=results, total=len(results), query=q)


@app.post("/search/image")
async def search_image(
    image_url: str = Query(..., description="Image URL"),
    city: Optional[str] = None,
    top_k: int = Query(default=10, ge=1, le=100),
) -> SearchResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    filters = {"city": city} if city else None
    results = await retriever.image_search(
        image_url=image_url, filters=filters, top_k=top_k
    )
    return SearchResponse(results=results, total=len(results), query=image_url)


@app.post("/ingest")
async def ingest(req: IngestRequest) -> Dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    try:
        retriever.add_travel_memories(req.items)
        return {"status": "ok", "ingested": len(req.items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/stats")
async def collection_stats() -> Dict[str, Any]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever.get_stats()

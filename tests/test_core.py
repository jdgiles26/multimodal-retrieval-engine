"""
Test suite for core retrieval engine
"""
import pytest
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llamaindex_engine import (
    Qwen3VLEmbedding,
    DashScopeReranker,
    LlamaIndexMultimodalRetriever,
    EmbeddingError,
)


@pytest.fixture
def api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        pytest.skip("DASHSCOPE_API_KEY not set")
    return key


@pytest.fixture
def embedder(api_key: str) -> Qwen3VLEmbedding:
    return Qwen3VLEmbedding(api_key=api_key)


@pytest.fixture
def reranker(api_key: str) -> DashScopeReranker:
    return DashScopeReranker(api_key=api_key)


@pytest.mark.asyncio
async def test_embed_text(embedder: Qwen3VLEmbedding) -> None:
    """Test text embedding"""
    embedding = await embedder._aget_text_embedding("test text")
    assert len(embedding) == 2560
    assert isinstance(embedding[0], float)


@pytest.mark.asyncio
async def test_embed_image(embedder: Qwen3VLEmbedding) -> None:
    """Test image embedding"""
    embedding = await embedder._aget_image_embedding(
        "https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"
    )
    assert len(embedding) == 2560
    assert isinstance(embedding[0], float)


@pytest.mark.asyncio
async def test_similarity(embedder: Qwen3VLEmbedding) -> None:
    """Test that similar texts produce high similarity"""
    import numpy as np

    emb1 = await embedder._aget_text_embedding("Tokyo Tower")
    emb2 = await embedder._aget_text_embedding("Tokyo Tower")

    sim = float(
        np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    )
    assert 0 <= sim <= 1
    assert sim > 0.95


def test_embedding_error() -> None:
    """Test that empty input raises EmbeddingError"""
    emb = Qwen3VLEmbedding(api_key="dummy")
    with pytest.raises(EmbeddingError):
        emb._get_multimodal_embedding("", "")


def test_clean_text_for_api() -> None:
    """Test emoji stripping in reranker"""
    cleaned = DashScopeReranker._clean_text_for_api("Hello world")
    assert cleaned == "Hello world"

    cleaned = DashScopeReranker._clean_text_for_api("")
    assert cleaned == "query"

"""
测试套件
"""
import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Qwen3VLEmbedder, Qwen3VLReranker, MilvusClient


@pytest.fixture
def embedder():
    return Qwen3VLEmbedder()


@pytest.fixture
def reranker():
    return Qwen3VLReranker()


@pytest.fixture
def milvus_client():
    client = MilvusClient()
    client.connect()
    return client


@pytest.mark.asyncio
async def test_embed_text(embedder):
    """测试文本编码"""
    embedding = await embedder.embed_text("测试文本")
    assert embedding.shape == (2560,)
    assert isinstance(embedding[0], (float,))


@pytest.mark.asyncio
async def test_embed_image(embedder):
    """测试图片编码"""
    # 使用示例图片
    embedding = await embedder.embed_image(
        "https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"
    )
    assert embedding.shape == (2560,)


@pytest.mark.asyncio
async def test_embed_fusion(embedder):
    """测试融合编码"""
    embedding = await embedder.embed_fusion(
        text="测试文本",
        images=["https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"]
    )
    assert embedding.shape == (2560,)


@pytest.mark.asyncio
async def test_similarity(embedder):
    """测试相似度计算"""
    emb1 = await embedder.embed_text("东京塔")
    emb2 = await embedder.embed_text("东京铁塔")
    
    sim = await embedder.compute_similarity(emb1, emb2)
    assert 0 <= sim <= 1
    assert sim > 0.5  # 相似文本应该有较高相似度


@pytest.mark.asyncio
async def test_rerank(reranker):
    """测试 Rerank"""
    query = {"text": "东京塔夜景"}
    candidates = [
        {"id": "1", "content": "东京塔夜景照片", "image": ""},
        {"id": "2", "content": "悉尼歌剧院", "image": ""},
    ]
    
    results = await reranker.rerank(query, candidates, top_k=2)
    assert len(results) == 2
    assert "score" in results[0]


def test_milvus_connection(milvus_client):
    """测试 Milvus 连接"""
    stats = milvus_client.get_stats()
    assert "name" in stats
    assert "num_entities" in stats


def test_filter_expr(milvus_client):
    """测试过滤表达式构建"""
    filters = {
        "city": "Tokyo",
        "tags": ["夜景", "汉服"],
        "date_from": "2026-03-01",
        "date_to": "2026-03-31"
    }
    
    expr = milvus_client.build_filter_expr(filters)
    assert "city" in expr
    assert "tags" in expr
    assert "date" in expr
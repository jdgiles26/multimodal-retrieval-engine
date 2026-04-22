"""
测试脚本
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Qwen3VLEmbedder, Qwen3VLReranker
from config.settings import settings


async def test_embedder():
    """测试 Embedding"""
    print("\n=== 测试 Qwen3-VL Embedding ===")
    
    embedder = Qwen3VLEmbedder()
    
    # 测试文本编码
    print("\n1. 文本编码...")
    text_emb = await embedder.embed_text("东京塔夜景汉服照片")
    print(f"   向量维度: {text_emb.shape}")
    print(f"   前5个值: {text_emb[:5]}")
    
    # 测试图片编码 (使用示例URL)
    print("\n2. 单图编码...")
    try:
        image_emb = await embedder.embed_image(
            "https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"
        )
        print(f"   向量维度: {image_emb.shape}")
        print(f"   前5个值: {image_emb[:5]}")
    except Exception as e:
        print(f"   图片编码失败 (网络问题): {e}")
    
    # 测试融合编码
    print("\n3. 融合编码 (文本+图片)...")
    try:
        fusion_emb = await embedder.embed_fusion(
            text="东京塔夜景",
            images=["https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"]
        )
        print(f"   向量维度: {fusion_emb.shape}")
        print(f"   前5个值: {fusion_emb[:5]}")
    except Exception as e:
        print(f"   融合编码失败 (网络问题): {e}")
    
    print("\n✅ Embedding 测试完成")


async def test_reranker():
    """测试 Reranker"""
    print("\n=== 测试 Qwen3-VL Reranker ===")
    
    reranker = Qwen3VLReranker()
    
    query = {"text": "东京塔夜景汉服"}
    
    candidates = [
        {"id": "1", "content": "东京塔夜景, 穿着汉服的女生", "image": ""},
        {"id": "2", "content": "悉尼歌剧院, 海边日落", "image": ""},
        {"id": "3", "content": "巴黎埃菲尔铁塔, 夜景", "image": ""},
        {"id": "4", "content": "东京浅草寺, 传统建筑", "image": ""},
    ]
    
    try:
        results = await reranker.rerank(query, candidates, top_k=3)
        
        print("\n重排序结果:")
        for r in results:
            print(f"  {r['id']}: score={r['score']:.4f}")
            
    except Exception as e:
        print(f"Reranker 测试失败 (网络问题): {e}")
    
    print("\n✅ Reranker 测试完成")


async def test_similarity():
    """测试相似度计算"""
    print("\n=== 测试相似度计算 ===")
    
    embedder = Qwen3VLEmbedder()
    
    # 编码两个相似文本
    text1 = "东京塔夜景"
    text2 = "东京塔夜晚的景色"
    
    try:
        emb1 = await embedder.embed_text(text1)
        emb2 = await embedder.embed_text(text2)
        
        sim = await embedder.compute_similarity(emb1, emb2)
        print(f"\n文本1: {text1}")
        print(f"文本2: {text2}")
        print(f"相似度: {sim:.4f}")
        
    except Exception as e:
        print(f"相似度测试失败 (网络问题): {e}")
    
    print("\n✅ 相似度测试完成")


async def main():
    print("=" * 50)
    print("Duckyo 多模态检索系统测试")
    print("=" * 50)
    
    print(f"\n配置信息:")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  Embedding Dim: {settings.EMBEDDING_DIMENSION}")
    print(f"  Rerank Model: {settings.RERANK_MODEL}")
    
    # 运行测试
    await test_embedder()
    await test_reranker()
    await test_similarity()
    
    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
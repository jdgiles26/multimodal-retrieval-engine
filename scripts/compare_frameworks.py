"""
三种框架对比测试
"""
import asyncio
import os
import sys
sys.path.insert(0, '/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval')

print("=" * 70)
print("🦊 多模态检索引擎 - 三种框架对比")
print("=" * 70)

# 测试数据
test_memories = [
    {"content": "东京塔夜景, 穿着汉服的fox girl", "city": "Tokyo", "date": "2026-03-06"},
    {"content": "悉尼歌剧院, 海港大桥", "city": "Sydney", "date": "2026-03-09"},
    {"content": "埃菲尔铁塔, 浪漫之都", "city": "Paris", "date": "2026-03-15"},
    {"content": "北极光, 极光爆发", "city": "Reykjavik", "date": "2026-03-20"},
]

api_key = os.getenv("DASHSCOPE_API_KEY", "sk-a766f04fac884533995ce54b5fe54979")

# ========== 方案1: 自定义版本 ==========
print("\n📦 方案1: 自定义版本")
print("-" * 70)

from core.embedder import Qwen3VLEmbedder

embedder = Qwen3VLEmbedder(api_key=api_key)

# 编码测试
import numpy as np
vectors = {}
for mem in test_memories:
    emb = asyncio.run(embedder.embed_text(mem["content"]))
    vectors[mem["content"]] = emb

# 搜索测试
query = "东京塔夜景"
query_emb = asyncio.run(embedder.embed_text(query))

results = []
for content, vec in vectors.items():
    sim = asyncio.run(embedder.compute_similarity(query_emb, vec))
    results.append((content, sim))

results.sort(key=lambda x: x[1], reverse=True)

print(f"查询: '{query}'")
for content, sim in results[:3]:
    print(f"  🎯 {sim:.3f} | {content[:30]}...")

print("✅ 自定义版本: 完全控制，灵活度高")

# ========== 方案2: LangChain 版本 ==========
print("\n📦 方案2: LangChain 版本")
print("-" * 70)

try:
    from core.langchain_engine import Qwen3VLEmbeddings
    
    embeddings = Qwen3VLEmbeddings(
        api_key=api_key,
        model="qwen3-vl-embedding",
        dimension=2560
    )
    
    # 编码测试
    texts = [mem["content"] for mem in test_memories]
    doc_embeddings = embeddings.embed_documents(texts)
    query_embedding = embeddings.embed_query(query)
    
    # 计算相似度
    results = []
    for i, (mem, emb) in enumerate(zip(test_memories, doc_embeddings)):
        sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        results.append((mem["content"], sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"查询: '{query}'")
    for content, sim in results[:3]:
        print(f"  🎯 {sim:.3f} | {content[:30]}...")
    
    print("✅ LangChain 版本: 生态丰富，标准化接口")
    
except Exception as e:
    print(f"⚠️  LangChain 版本需要 Milvus 服务: {e}")

# ========== 方案3: LlamaIndex 版本 ==========
print("\n📦 方案3: LlamaIndex 版本")
print("-" * 70)

try:
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    
    retriever = LlamaIndexMultimodalRetriever(
        api_key=api_key,
        milvus_uri="http://localhost:19530"
    )
    
    # 添加数据
    retriever.add_travel_memories(test_memories)
    
    # 搜索
    results = retriever.search(query, top_k=3)
    
    print(f"查询: '{query}'")
    for r in results:
        print(f"  🎯 {r['score']:.3f} | {r['content'][:30]}...")
    
    print("✅ LlamaIndex 版本: 原生多模态，代码简洁")
    
except Exception as e:
    print(f"⚠️  LlamaIndex 版本需要 Milvus 服务: {e}")

# ========== 对比总结 ==========
print("\n" + "=" * 70)
print("📊 三种方案对比总结")
print("=" * 70)

comparison = """
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│     特性        │   自定义    │  LangChain  │ LlamaIndex  │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ 代码量          │    中等     │    较少     │    最少     │
│ 多模态支持      │    ⭐⭐⭐    │    ⭐⭐⭐    │   ⭐⭐⭐⭐⭐  │
│ 生态集成        │    ⭐⭐     │   ⭐⭐⭐⭐⭐  │   ⭐⭐⭐⭐   │
│ 学习曲线        │    低      │    中等     │    中等     │
│ 生产就绪        │    ⭐⭐⭐    │   ⭐⭐⭐⭐   │   ⭐⭐⭐⭐   │
│ 灵活度          │    高      │    中等     │    中等     │
└─────────────────┴─────────────┴─────────────┴─────────────┘

推荐:
• 快速原型: 自定义版本 ✅
• 生产环境: LangChain 或 LlamaIndex ✅
• 多模态RAG: LlamaIndex ✅ (原生支持)
• 复杂AI应用: LangChain ✅ (生态最丰富)
"""

print(comparison)

print("=" * 70)
print("✅ 对比测试完成!")
print("=" * 70)
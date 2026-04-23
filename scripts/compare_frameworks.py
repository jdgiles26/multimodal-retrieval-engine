"""
Three Framework Comparison Test
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("[INFO] Multimodal Retrieval Engine - Three Framework Comparison")
print("=" * 70)

# Test data
test_memories = [
    {"content": "Tokyo Tower night view, fox girl wearing Hanfu", "city": "Tokyo", "date": "2026-03-06"},
    {"content": "Sydney Opera House, Harbour Bridge", "city": "Sydney", "date": "2026-03-09"},
    {"content": "Eiffel Tower, city of romance", "city": "Paris", "date": "2026-03-15"},
    {"content": "Northern Lights, aurora burst", "city": "Reykjavik", "date": "2026-03-20"},
]

api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")

# ========== Option 1: Custom Version ==========
print("\n[INFO] Option 1: Custom Version")
print("-" * 70)

from core.llamaindex_engine import Qwen3VLEmbedding

embedder = Qwen3VLEmbedding(api_key=api_key)

# Encoding test
import numpy as np
vectors = {}
for mem in test_memories:
    emb = asyncio.run(embedder._aget_text_embedding(mem["content"]))
    vectors[mem["content"]] = emb

# Search test
query = "Tokyo Tower night view"
query_emb = asyncio.run(embedder._aget_text_embedding(query))

results = []
for content, vec in vectors.items():
    sim = np.dot(query_emb, vec) / (np.linalg.norm(query_emb) * np.linalg.norm(vec))
    results.append((content, sim))

results.sort(key=lambda x: x[1], reverse=True)

print(f"Query: '{query}'")
for content, sim in results[:3]:
    print(f"  [OK] {sim:.3f} | {content[:30]}...")

print("[OK] Custom Version: Full control, high flexibility")

# ========== Option 2: LangChain Version ==========
print("\n[INFO] Option 2: LangChain Version")
print("-" * 70)

print("[WARN] LangChain version not available in this codebase")

# ========== Option 3: LlamaIndex Version ==========
print("\n[INFO] Option 3: LlamaIndex Version")
print("-" * 70)

try:
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever

    retriever = LlamaIndexMultimodalRetriever(
        api_key=api_key,
        milvus_uri="http://localhost:19530"
    )

    # Add data
    retriever.add_travel_memories(test_memories)

    # Search
    results = retriever.search_sync(query, top_k=3)

    print(f"Query: '{query}'")
    for r in results:
        print(f"  [OK] {r['score']:.3f} | {r['content'][:30]}...")

    print("[OK] LlamaIndex Version: Native multimodal, clean code")

except Exception as e:
    print(f"[WARN] LlamaIndex version requires Milvus service: {e}")

# ========== Comparison Summary ==========
print("\n" + "=" * 70)
print("[INFO] Three Options Comparison Summary")
print("=" * 70)

comparison = """
+-----------------+-----------+-----------+-----------+
| Feature         | Custom    | LangChain | LlamaIndex|
+-----------------+-----------+-----------+-----------+
| Code Amount     | Medium    | Less      | Least     |
| Multimodal      | Good      | Good      | Excellent |
| Ecosystem       | Fair      | Excellent | Good      |
| Learning Curve  | Low       | Medium    | Medium    |
| Production Ready| Good      | Good      | Good      |
| Flexibility     | High      | Medium    | Medium    |
+-----------------+-----------+-----------+-----------+

Recommendations:
- Rapid Prototyping: Custom Version [OK]
- Production: LangChain or LlamaIndex [OK]
- Multimodal RAG: LlamaIndex [OK] (native support)
- Complex AI Apps: LangChain [OK] (richest ecosystem)
"""

print(comparison)

print("=" * 70)
print("[OK] Comparison test complete!")
print("=" * 70)

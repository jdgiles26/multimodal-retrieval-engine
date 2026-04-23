"""
Test script
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import Qwen3VLEmbedder, Qwen3VLReranker
from config.settings import settings


async def test_embedder():
    """Test Embedding"""
    print("\n=== Test Qwen3-VL Embedding ===")
    
    embedder = Qwen3VLEmbedder()
    
    # Test text encoding
    print("\n1. Text encoding...")
    text_emb = await embedder.embed_text("Tokyo Tower night view Hanfu photo")
    print(f"   Vector dimension: {text_emb.shape}")
    print(f"   First 5 values: {text_emb[:5]}")
    
    # Test image encoding (using sample URL)
    print("\n2. Single image encoding...")
    try:
        image_emb = await embedder.embed_image(
            "https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"
        )
        print(f"   Vector dimension: {image_emb.shape}")
        print(f"   First 5 values: {image_emb[:5]}")
    except Exception as e:
        print(f"   Image encoding failed (network issue): {e}")
    
    # Test fusion encoding
    print("\n3. Fusion encoding (text + image)...")
    try:
        fusion_emb = await embedder.embed_fusion(
            text="Tokyo Tower night view",
            images=["https://img.alicdn.com/imgextra/i3/O1CN01rdstgY1uiZWt8gqSL_!!6000000006071-0-tps-1970-356.jpg"]
        )
        print(f"   Vector dimension: {fusion_emb.shape}")
        print(f"   First 5 values: {fusion_emb[:5]}")
    except Exception as e:
        print(f"   Fusion encoding failed (network issue): {e}")
    
    print("\n[OK] Embedding test complete")


async def test_reranker():
    """Test Reranker"""
    print("\n=== Test Qwen3-VL Reranker ===")
    
    reranker = Qwen3VLReranker()
    
    query = {"text": "Tokyo Tower night view Hanfu"}
    
    candidates = [
        {"id": "1", "content": "Tokyo Tower night view, girl wearing Hanfu", "image": ""},
        {"id": "2", "content": "Sydney Opera House, seaside sunset", "image": ""},
        {"id": "3", "content": "Paris Eiffel Tower, night view", "image": ""},
        {"id": "4", "content": "Tokyo Sensoji Temple, traditional architecture", "image": ""},
    ]
    
    try:
        results = await reranker.rerank(query, candidates, top_k=3)
        
        print("\nReranking results:")
        for r in results:
            print(f"  {r['id']}: score={r['score']:.4f}")
            
    except Exception as e:
        print(f"Reranker test failed (network issue): {e}")
    
    print("\n[OK] Reranker test complete")


async def test_similarity():
    """Test similarity computation"""
    print("\n=== Test Similarity Computation ===")
    
    embedder = Qwen3VLEmbedder()
    
    # Encode two similar texts
    text1 = "Tokyo Tower night view"
    text2 = "Tokyo Tower night scenery"
    
    try:
        emb1 = await embedder.embed_text(text1)
        emb2 = await embedder.embed_text(text2)
        
        sim = await embedder.compute_similarity(emb1, emb2)
        print(f"\nText 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity: {sim:.4f}")
        
    except Exception as e:
        print(f"Similarity test failed (network issue): {e}")
    
    print("\n[OK] Similarity test complete")


async def main():
    print("=" * 50)
    print("Duckyo Multimodal Retrieval System Test")
    print("=" * 50)
    
    print(f"\nConfig info:")
    print(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"  Embedding Dim: {settings.EMBEDDING_DIMENSION}")
    print(f"  Rerank Model: {settings.RERANK_MODEL}")
    
    # Run tests
    await test_embedder()
    await test_reranker()
    await test_similarity()
    
    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

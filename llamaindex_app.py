"""
LlamaIndex Multimodal Retrieval System - Thin CLI wrapper
Imports all engine logic from core.llamaindex_engine to avoid duplication.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core.llamaindex_engine import (
    LlamaIndexMultimodalRetriever,
    get_llamaindex_retriever,
)

# Re-export for backward compatibility
create_retriever = get_llamaindex_retriever


def _create_retriever_from_env() -> LlamaIndexMultimodalRetriever:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")
    return LlamaIndexMultimodalRetriever(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="LlamaIndex Multimodal Retrieval System")
    parser.add_argument(
        "--action",
        choices=["init", "add", "search", "search-image", "stats"],
        required=True,
    )
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--image", type=str, help="Image URL")
    parser.add_argument("--top-k", type=int, default=10, help="Return count")
    parser.add_argument("--data", type=str, help="Data JSON file")

    args = parser.parse_args()
    retriever = _create_retriever_from_env()

    if args.action == "init":
        print("[INFO] Initializing collection...")
        retriever.create_index([])
        print("[OK] Initialization complete!")

    elif args.action == "add":
        if not args.data:
            print("[ERROR] Please specify --data argument")
            sys.exit(1)
        with open(args.data, "r") as f:
            memories = json.load(f)
        retriever.add_travel_memories(memories)
        print(f"[OK] Added {len(memories)} memories!")

    elif args.action == "search":
        if not args.query:
            print("[ERROR] Please specify --query argument")
            sys.exit(1)
        results = retriever.search_sync(args.query, top_k=args.top_k)
        print(f"\n[INFO] Search: '{args.query}'")
        print(f"[INFO] Found {len(results)} results\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f}")
            print(f"    Content: {r['content'][:50]}...")
            print(f"    City: {r['metadata'].get('city', 'N/A')}")
            print(f"    Date: {r['metadata'].get('date', 'N/A')}")
            print()

    elif args.action == "search-image":
        if not args.image:
            print("[ERROR] Please specify --image argument")
            sys.exit(1)
        results = retriever.search_by_image(args.image, top_k=args.top_k)
        print(f"\n[INFO] Image search: {args.image}")
        print(f"[INFO] Found {len(results)} results\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.3f}")
            print(f"    Content: {r['content'][:50]}...")
            print(f"    City: {r['metadata'].get('city', 'N/A')}")
            print()

    elif args.action == "stats":
        stats = retriever.get_stats()
        print("\n[INFO] Collection stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI Tool - Multimodal Retrieval Command Line Interface

Usage:
    python -m cli search "Tokyo tower night view"
    python -m cli search-image https://example.com/photo.jpg
    python -m cli ingest --dir /path/to/photos
    python -m cli stats
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from core import get_engine
from config.settings import get_settings


def print_json(data: dict) -> None:
    """Pretty print JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def cmd_search(args: argparse.Namespace) -> None:
    """Search command"""
    engine = get_engine()

    results = await engine.search(
        query_text=args.query,
        filters={"city": args.city} if args.city else None,
        top_k=args.top_k,
        use_rerank=not args.no_rerank
    )

    print(f"\n[INFO] Search: {args.query}")
    print(f"[INFO] Found {len(results)} results\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.get('file_url', 'N/A')}")
        print(f"    Content: {r.get('content', 'N/A')[:50]}...")
        print(f"    City: {r.get('city', 'N/A')} | Date: {r.get('date', 'N/A')}")
        print(f"    Vector score: {r.get('vector_score', 0):.4f} | Rerank score: {r.get('rerank_score', 0):.4f}")
        print()


async def cmd_search_image(args: argparse.Namespace) -> None:
    """Image-to-image search command"""
    engine = get_engine()

    results = await engine.image_search(
        image_url=args.image_url,
        filters={"city": args.city} if args.city else None,
        top_k=args.top_k
    )

    print(f"\n[INFO] Image search: {args.image_url}")
    print(f"[INFO] Found {len(results)} results\n")

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.get('file_url', 'N/A')}")
        print(f"    Similarity: {r.get('rerank_score', 0):.4f}")
        print()


async def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest command"""
    from scripts.ingest_travel_data import scan_photos_dir

    engine = get_engine()

    photos = scan_photos_dir(args.dir)
    print(f"[INFO] Scanned {len(photos)} photos")

    if args.dry_run:
        print("\n[DRY RUN] Preview first 3:")
        for p in photos[:3]:
            print(f"  - {p['file_path']}")
        return

    await engine.ingest(photos, batch_size=args.batch_size)
    print(f"[OK] Successfully imported {len(photos)} photos")


async def cmd_stats(args: argparse.Namespace) -> None:
    """Stats command"""
    from pymilvus import connections, utility
    settings = get_settings()
    connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
    stats = {
        "collection": settings.MILVUS_COLLECTION,
        "exists": utility.has_collection(settings.MILVUS_COLLECTION),
    }
    if stats["exists"]:
        from pymilvus import Collection
        col = Collection(settings.MILVUS_COLLECTION)
        stats["count"] = col.num_entities
    print("\n[INFO] Collection stats:")
    print_json(stats)


async def cmd_init(args: argparse.Namespace) -> None:
    """Init command"""
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
    settings = get_settings()
    connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

    print(f"[INFO] Creating collection: {settings.MILVUS_COLLECTION}")

    fields = [
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="video_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="mood", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="outfit", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="media_type", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dimension),
    ]
    schema = CollectionSchema(fields, description="Multimodal retrieval collection")
    collection = Collection(name=settings.MILVUS_COLLECTION, schema=schema)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 128},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("[OK] Initialization complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multimodal-retrieval",
        description="Multimodal Retrieval CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Text-to-image search")
    search_parser.add_argument("query", help="Search text")
    search_parser.add_argument("--city", help="City filter")
    search_parser.add_argument("--top-k", type=int, default=10, help="Return count")
    search_parser.add_argument("--no-rerank", action="store_true", help="Disable rerank")

    img_parser = subparsers.add_parser("search-image", help="Image-to-image search")
    img_parser.add_argument("image_url", help="Image URL")
    img_parser.add_argument("--city", help="City filter")
    img_parser.add_argument("--top-k", type=int, default=10, help="Return count")

    ingest_parser = subparsers.add_parser("ingest", help="Import data")
    ingest_parser.add_argument("--dir", required=True, help="Photo directory")
    ingest_parser.add_argument("--batch-size", type=int, default=20, help="Batch size")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Preview mode")

    subparsers.add_parser("stats", help="View statistics")

    init_parser = subparsers.add_parser("init", help="Initialize collection")
    init_parser.add_argument("--dimension", type=int, default=2560, help="Vector dimension")
    init_parser.add_argument("--force", action="store_true", help="Force rebuild")

    return parser


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "search":
        await cmd_search(args)
    elif args.command == "search-image":
        await cmd_search_image(args)
    elif args.command == "ingest":
        await cmd_ingest(args)
    elif args.command == "stats":
        await cmd_stats(args)
    elif args.command == "init":
        await cmd_init(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

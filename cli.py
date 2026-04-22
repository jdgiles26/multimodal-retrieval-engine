#!/usr/bin/env python3
"""
CLI 工具 - 多模态检索命令行

用法:
    python -m cli search "东京塔夜景"
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

from core import get_engine, get_milvus_client
from config.settings import settings


def print_json(data):
    """美化打印 JSON"""
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def cmd_search(args):
    """搜索命令"""
    engine = get_engine()
    
    results = await engine.search(
        query_text=args.query,
        filters={"city": args.city} if args.city else None,
        top_k=args.top_k,
        use_rerank=not args.no_rerank
    )
    
    print(f"\n🔍 搜索: {args.query}")
    print(f"📊 找到 {len(results)} 条结果\n")
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.get('file_url', 'N/A')}")
        print(f"    内容: {r.get('content', 'N/A')[:50]}...")
        print(f"    城市: {r.get('city', 'N/A')} | 日期: {r.get('date', 'N/A')}")
        print(f"    向量分: {r.get('vector_score', 0):.4f} | 精排分: {r.get('rerank_score', 0):.4f}")
        print()


async def cmd_search_image(args):
    """图搜图命令"""
    engine = get_engine()
    
    results = await engine.image_search(
        image_url=args.image_url,
        filters={"city": args.city} if args.city else None,
        top_k=args.top_k
    )
    
    print(f"\n🖼️  图搜图: {args.image_url}")
    print(f"📊 找到 {len(results)} 条结果\n")
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.get('file_url', 'N/A')}")
        print(f"    相似度: {r.get('rerank_score', 0):.4f}")
        print()


async def cmd_ingest(args):
    """导入命令"""
    from scripts.ingest_travel_data import scan_photos_dir
    
    engine = get_engine()
    
    if args.dir:
        photos = scan_photos_dir(args.dir)
        print(f"📁 扫描到 {len(photos)} 张照片")
        
        if args.dry_run:
            print("\n[DRY RUN] 预览前3张:")
            for p in photos[:3]:
                print(f"  - {p['file_path']}")
            return
        
        await engine.ingest(photos, batch_size=args.batch_size)
        print(f"✅ 成功导入 {len(photos)} 张照片")
    else:
        print("❌ 请指定 --dir 参数")


async def cmd_stats(args):
    """统计命令"""
    client = get_milvus_client()
    client.connect()
    
    stats = client.get_stats()
    print("\n📊 Collection 统计:")
    print_json(stats)


async def cmd_init(args):
    """初始化命令"""
    client = get_milvus_client()
    client.connect()
    
    print(f"🚀 创建 Collection: {settings.MILVUS_COLLECTION}")
    client.create_collection(
        dimension=args.dimension,
        drop_existing=args.force
    )
    print("✅ 初始化完成!")


async def main():
    parser = argparse.ArgumentParser(
        prog="multimodal-retrieval",
        description="Duckyo 多模态检索 CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="文搜图")
    search_parser.add_argument("query", help="搜索文本")
    search_parser.add_argument("--city", help="城市过滤")
    search_parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    search_parser.add_argument("--no-rerank", action="store_true", help="禁用Rerank")
    
    # search-image 命令
    img_parser = subparsers.add_parser("search-image", help="图搜图")
    img_parser.add_argument("image_url", help="图片URL")
    img_parser.add_argument("--city", help="城市过滤")
    img_parser.add_argument("--top-k", type=int, default=10, help="返回数量")
    
    # ingest 命令
    ingest_parser = subparsers.add_parser("ingest", help="导入数据")
    ingest_parser.add_argument("--dir", required=True, help="照片目录")
    ingest_parser.add_argument("--batch-size", type=int, default=20, help="批量大小")
    ingest_parser.add_argument("--dry-run", action="store_true", help="预览模式")
    
    # stats 命令
    subparsers.add_parser("stats", help="查看统计")
    
    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化Collection")
    init_parser.add_argument("--dimension", type=int, default=2560, help="向量维度")
    init_parser.add_argument("--force", action="store_true", help="强制重建")
    
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
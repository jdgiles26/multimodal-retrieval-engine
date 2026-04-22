#!/usr/bin/env python3
"""
示例: 批量导入旅途数据

用法:
    python scripts/ingest_travel_data.py --photos-dir /path/to/photos --memories /path/to/memories.md
"""
import os
import sys
import asyncio
import argparse
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import get_engine, get_milvus_client
from config.settings import settings


# 旅途城市映射
CITY_MAPPING = {
    "tokyo": {"city": "Tokyo", "country": "Japan", "city_cn": "东京"},
    "sydney": {"city": "Sydney", "country": "Australia", "city_cn": "悉尼"},
    "cape-town": {"city": "Cape Town", "country": "South Africa", "city_cn": "开普敦"},
    "paris": {"city": "Paris", "country": "France", "city_cn": "巴黎"},
    "iceland": {"city": "Reykjavik", "country": "Iceland", "city_cn": "冰岛"},
    "new-york": {"city": "New York", "country": "USA", "city_cn": "纽约"},
}


def scan_photos_dir(photos_dir: str) -> list:
    """扫描照片目录"""
    photos = []
    photos_path = Path(photos_dir)
    
    if not photos_path.exists():
        logger.warning(f"Photos directory not found: {photos_dir}")
        return photos
    
    # 支持的图片格式
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    for file_path in photos_path.rglob("*"):
        if file_path.suffix.lower() in extensions:
            # 从路径中提取城市信息
            parts = file_path.parts
            city_key = None
            for part in parts:
                part_lower = part.lower()
                for key in CITY_MAPPING.keys():
                    if key in part_lower:
                        city_key = key
                        break
                if city_key:
                    break
            
            # 从文件名提取日期
            date_str = file_path.stem[:10]  # YYYY-MM-DD
            
            photos.append({
                "file_path": str(file_path),
                "file_url": f"file://{file_path}",
                "city_key": city_key,
                "date": date_str,
                "filename": file_path.name
            })
    
    return photos


def generate_description(photo_info: dict) -> str:
    """根据照片信息生成描述"""
    city_info = CITY_MAPPING.get(photo_info["city_key"], {})
    city = city_info.get("city_cn", photo_info["city_key"] or "未知")
    date = photo_info["date"]
    
    # 从文件名推断内容
    filename = photo_info["filename"].lower()
    content = "旅行照片"
    
    if "selfie" in filename or "自拍" in filename:
        content = "自拍"
    if "food" in filename or "美食" in filename:
        content = "美食"
    if "night" in filename or "夜景" in filename:
        content = "夜景"
    if "sea" in filename or "海" in filename:
        content = "海景"
    if "mountain" in filename or "山" in filename:
        content = "山景"
    
    return f"{city} {content} {date}"


async def ingest_photos(engine: MultimodalRetrievalEngine, photos: list, dry_run: bool = False):
    """导入照片数据"""
    logger.info(f"Found {len(photos)} photos")
    
    items = []
    for i, photo in enumerate(tqdm(photos, desc="Preparing data")):
        city_info = CITY_MAPPING.get(photo["city_key"], {})
        
        # 计算第几天
        try:
            date = datetime.strptime(photo["date"], "%Y-%m-%d")
            trip_start = datetime(2026, 3, 6)
            day_of_trip = (date - trip_start).days + 1
        except:
            day_of_trip = 0
        
        item = {
            "id": f"photo_{i:04d}",
            "type": "image",
            "content": generate_description(photo),
            "file_url": photo["file_url"],
            "city": city_info.get("city", ""),
            "country": city_info.get("country", ""),
            "date": photo["date"],
            "day_of_trip": day_of_trip,
            "tags": ["旅行", city_info.get("city_cn", "")],
            "mood": "开心",
            "outfit": "休闲",
            "source": "local_photos"
        }
        
        items.append(item)
    
    if dry_run:
        logger.info(f"[DRY RUN] Would ingest {len(items)} items")
        # 打印前3个示例
        for item in items[:3]:
            logger.info(f"  {item}")
        return
    
    # 批量入库
    await engine.ingest(items, batch_size=20)
    logger.info(f"Successfully ingested {len(items)} photos")


async def main():
    parser = argparse.ArgumentParser(description="Import travel data")
    parser.add_argument("--photos-dir", type=str, help="Photos directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--create-collection", action="store_true", help="Create collection first")
    
    args = parser.parse_args()
    
    # 连接 Milvus
    milvus_client = get_milvus_client()
    milvus_client.connect()
    
    # 创建 Collection
    if args.create_collection:
        milvus_client.create_collection(dimension=2560, drop_existing=False)
    
    # 获取引擎
    engine = get_engine()
    
    # 扫描照片
    if args.photos_dir:
        photos = scan_photos_dir(args.photos_dir)
        await ingest_photos(engine, photos, dry_run=args.dry_run)
    else:
        logger.info("No photos directory specified")
        
        # 测试: 打印城市映射
        logger.info("Available cities:")
        for key, info in CITY_MAPPING.items():
            logger.info(f"  {key}: {info}")


if __name__ == "__main__":
    asyncio.run(main())
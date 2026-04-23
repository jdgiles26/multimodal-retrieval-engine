#!/usr/bin/env python3
"""
Example: Batch import travel data

Usage:
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import get_engine, get_milvus_client
from config.settings import settings


# Travel city mapping
CITY_MAPPING = {
    "tokyo": {"city": "Tokyo", "country": "Japan", "city_cn": "Tokyo"},
    "sydney": {"city": "Sydney", "country": "Australia", "city_cn": "Sydney"},
    "cape-town": {"city": "Cape Town", "country": "South Africa", "city_cn": "Cape Town"},
    "paris": {"city": "Paris", "country": "France", "city_cn": "Paris"},
    "iceland": {"city": "Reykjavik", "country": "Iceland", "city_cn": "Iceland"},
    "new-york": {"city": "New York", "country": "USA", "city_cn": "New York"},
}


def scan_photos_dir(photos_dir: str) -> list:
    """Scan photos directory"""
    photos = []
    photos_path = Path(photos_dir)
    
    if not photos_path.exists():
        logger.warning(f"Photos directory not found: {photos_dir}")
        return photos
    
    # Supported image formats
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    for file_path in photos_path.rglob("*"):
        if file_path.suffix.lower() in extensions:
            # Extract city info from path
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
            
            # Extract date from filename
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
    """Generate description from photo info"""
    city_info = CITY_MAPPING.get(photo_info["city_key"], {})
    city = city_info.get("city_cn", photo_info["city_key"] or "Unknown")
    date = photo_info["date"]
    
    # Infer content from filename
    filename = photo_info["filename"].lower()
    content = "travel photo"
    
    if "selfie" in filename:
        content = "selfie"
    if "food" in filename:
        content = "food"
    if "night" in filename:
        content = "night view"
    if "sea" in filename:
        content = "seascape"
    if "mountain" in filename:
        content = "mountain view"
    
    return f"{city} {content} {date}"


async def ingest_photos(engine: MultimodalRetrievalEngine, photos: list, dry_run: bool = False):
    """Import photo data"""
    logger.info(f"Found {len(photos)} photos")
    
    items = []
    for i, photo in enumerate(tqdm(photos, desc="Preparing data")):
        city_info = CITY_MAPPING.get(photo["city_key"], {})
        
        # Calculate day number
        try:
            date = datetime.strptime(photo["date"], "%Y-%m-%d")
            trip_start = datetime(2026, 3, 6)
            day_of_trip = (date - trip_start).days + 1
        except (ValueError, TypeError):
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
            "tags": ["travel", city_info.get("city_cn", "")],
            "mood": "happy",
            "outfit": "casual",
            "source": "local_photos"
        }
        
        items.append(item)
    
    if dry_run:
        logger.info(f"[DRY RUN] Would ingest {len(items)} items")
        # Print first 3 examples
        for item in items[:3]:
            logger.info(f"  {item}")
        return
    
    # Batch ingest
    await engine.ingest(items, batch_size=20)
    logger.info(f"Successfully ingested {len(items)} photos")


async def main():
    parser = argparse.ArgumentParser(description="Import travel data")
    parser.add_argument("--photos-dir", type=str, help="Photos directory")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--create-collection", action="store_true", help="Create collection first")
    
    args = parser.parse_args()
    
    # Connect to Milvus
    milvus_client = get_milvus_client()
    milvus_client.connect()
    
    # Create Collection
    if args.create_collection:
        milvus_client.create_collection(dimension=2560, drop_existing=False)
    
    # Get engine
    engine = get_engine()
    
    # Scan photos
    if args.photos_dir:
        photos = scan_photos_dir(args.photos_dir)
        await ingest_photos(engine, photos, dry_run=args.dry_run)
    else:
        logger.info("No photos directory specified")
        
        # Test: print city mapping
        logger.info("Available cities:")
        for key, info in CITY_MAPPING.items():
            logger.info(f"  {key}: {info}")


if __name__ == "__main__":
    asyncio.run(main())

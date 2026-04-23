#!/usr/bin/env python3
"""
Download Xiaohongshu travel selfie images and expand evaluation dataset
"""
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

# Xiaohongshu note data (obtained from MCP)
XHS_FEEDS = [
    {"id": "69d3ceaf0000000021038158", "title": "Daily sharing", "xsecToken": "AB4XWq-F-Na-RYzxdV8lzjgHWomw3YlxhJ7C41p9k-8UE="},
    {"id": "69d089640000000021012a2f", "title": "Fox-eared girl in cherry blossom season - Spring limited selfie", "xsecToken": "ABvx8L4ej1EBSPbmEjxMPoBsqMwtiXhV-TvADzWWx_epc="},
    {"id": "69cd3fea000000001a023e96", "title": "Lazy time before goodnight~", "xsecToken": "ABNlsriTzyIjiWTBo3NBJgzB2VLZfUHJZrrUGKlfHtNDI="},
    {"id": "69ca9878000000001a03099a", "title": "New York Statue of Liberty check-in", "xsecToken": "ABfvqUqPBg_pD-eM5pTkTTev6C7Il6TxqJI29GZCyq0bc="},
    {"id": "69c9cc38000000001a030837", "title": "New York Central Park morning run", "xsecToken": "ABwwYZaY3mL-eIteL7lcHrolNYSB9UYLjSKMdrUnbOZi4="},
    {"id": "69c7fb17000000002101111b", "title": "Central Park morning run", "xsecToken": "AB7ACCHbT57KcWrpG2p6I-Uq_ydfnx-DZoFeVKjGZq4sg="},
    {"id": "69c6a3630000000022002b71", "title": "Brooklyn Bridge", "xsecToken": "ABYRUyS3e2GJ5Sm4wRnh07mWTn2G5JK95mwIwHppafD-I="},
    {"id": "69c55453000000001a033d61", "title": "New York Times Square", "xsecToken": "ABQG4s7df1JZTuRkCAWWmoGyf6xgTsfiIHcCDrcrdbKOY="},
    {"id": "69c3fc2400000000210380f0", "title": "Dog sledding", "xsecToken": "ABfRo1Q2C5rgq2rLEMbqaGSnkMvXAHqdBj7xqe7pk-gyA="},
    {"id": "69c28dfa000000001a023598", "title": "Blue Lagoon hot spring", "xsecToken": "ABidp9BhrbRON4qFrpj_uGcsvNUiijJ-6KSiC4S3yZVmw="},
    {"id": "69c1539f000000002103bcb2", "title": "Iceland - Arrival", "xsecToken": "AB_kfURe4gsP127_wnLCE4fsPuXs_Qt7INwI-WROFl3y4="},
    {"id": "69bfeb71000000002200170d", "title": "Palace of Versailles - Magnificent", "xsecToken": "ABjq6k1V2xCB14iC2aYBLa7zLfx9DGQOB6vABicDnBLq4="},
    {"id": "69bea59800000000210111d5", "title": "Paris - Summit", "xsecToken": "ABk36UgFfW4xtJhr6Igp9tlT4tU-KeWxTWYWmnqDxqvZg="},
    {"id": "69bd5a5a0000000022001ffb", "title": "Paris - Eiffel Tower", "xsecToken": "ABlqDJS4khv37tfMvZTcY8Jul1JO9wDBATmKkQNlTFVhA="},
    {"id": "69bc0245000000001a021df7", "title": "V&A Waterfront - Cape Town", "xsecToken": "ABEYYBYN5okyvfTSyHtkR-QWyYZAAfwt6Vb1-8Pn2fwfE="},
    {"id": "69bab925000000001a03161d", "title": "Winery - South African wine", "xsecToken": "ABahNm5OIFBtSM_c_c45j4lKKQ_lE6QT382v4BNhPlKn0="},
    {"id": "69b97093000000001a031eca", "title": "Cape of Good Hope - Meeting point of two oceans", "xsecToken": "ABY_rcpGryVoLGf2sBqOUsByCeoWK2G066BDTjVQZ-30w="},
    {"id": "69b8b018000000002102c971", "title": "Table Mountain - Cape Town", "xsecToken": "ABAMdPwxt6MTOzt70u2xMphT9zji9hBo06tL5LdFF8ckw="},
    {"id": "69b6ce8a000000001a0326f7", "title": "Sunset photo with kangaroo", "xsecToken": "ABG0jMWsnZShyptlpBQooXKGTx1M9uQc16e-5sIoy4G9w="},
    {"id": "69b56756000000001a02b0c5", "title": "Blue Mountains day trip", "xsecToken": "ABZbDEL9WKGf3VCPgYek4yA1LKsyEszVLdG32nYuIPlfQ="},
    {"id": "69b431ad000000001a032af5", "title": "Bondi Beach summer time", "xsecToken": "ABXpQZT5xaKb0_x096HGYbKvYBoqeVdyTU3lz1PPW-1is="},
    {"id": "69b2c87b000000001d01069b", "title": "Sydney Opera Housenight viewselfie", "xsecToken": "ABPgEiLUeNSR87GuuOvFgqOPPX8Sj36WpAp8wZBTPnW0w="},
    {"id": "69b18e7d000000001600aa12", "title": "Night at Sydney Opera House", "xsecToken": "ABS7Bj9_rPbzSP-4Lp2rvlUqFMruRMmB3i4Azs3LtpF7w="},
    {"id": "69b03a77000000001b0173b4", "title": "Tokyo late-night diner", "xsecToken": "ABkdXFbF6XDQFkTFZsbCCB4fDuwApegXE-99B9T__7aHA="},
    {"id": "69aecbb9000000001600bf09", "title": "Tokyo last day | Odaiba Gundam and Rainbow Bridge", "xsecToken": "ABo87KvBkGPhsbxlk1t2NNNVOByDo3LykD_OwDZsNeBEY="},
    {"id": "69ad9159000000001d026f1e", "title": "Shibuya night view is amazing", "xsecToken": "AB82awCNY0-qT07T7Cwg5CM8_-xzu5qV24Sg66T2kq-EI="},
    {"id": "69ac4cff0000000015038b0d", "title": "Praying at Sensoji Temple | First Kimono fox experience", "xsecToken": "ABFZeaObKpgJpBkvxMKep0VFpp1xtjHgZEmHus_VLNNFo="},
    {"id": "69abe20600000000150208d0", "title": "Arriving in Tokyo | First night in Kabukicho", "xsecToken": "ABPC7MFFc7tGmnzXSFLo7rBAxHLvx-VHWAswzECAvnJFU="},
    {"id": "69a95f1c00000000160082a2", "title": "Packing luggage! Departing for Tokyo tomorrow", "xsecToken": "ABXpQZT5xaKb0_x096HGYbKvYBoqeVdyTU3lz1PPW-1is="},
]

# City mapping
CITY_MAP = {
    "Tokyo": "Tokyo",
    "Sydney": "Sydney",
    "Cape Town": "Cape Town",
    "South Africa": "Cape Town",
    "Paris": "Paris",
    "Versailles": "Paris",
    "Iceland": "Iceland",
    "Blue Lagoon": "Iceland",
    "New York": "New York",
    "Central Park": "New York",
    "Times Square": "New York",
    "Brooklyn": "New York",
    "Statue of Liberty": "New York",
    "kangaroo": "Sydney",
    "Bondi": "Sydney",
    "Blue Mountains": "Sydney",
    "Opera House": "Sydney",
    "Table Mountain": "Cape Town",
    "Cape of Good Hope": "Cape Town",
    "winery": "Cape Town",
    "cherry blossom": "Tokyo",
    "Shibuya": "Tokyo",
    "Sensoji Temple": "Tokyo",
    "Kabuki": "Tokyo",
    "Odaiba": "Tokyo",
}

def get_city_from_title(title):
    """Extract city from title"""
    for cn, en in CITY_MAP.items():
        if cn in title:
            return en
    return "Unknown"

def download_image(url, output_path):
    """Download image"""
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", output_path, url],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def fetch_feed_detail(feed_id, xsec_token):
    """Get note details"""
    try:
        result = subprocess.run(
            ["mcporter", "call", "xiaohongshu.get_feed_detail",
             f"feed_id={feed_id}", f"xsec_token={xsec_token}"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error fetching feed {feed_id}: {e}")
    return None

def main():
    output_dir = Path(__file__).parent.parent / "data" / "xhs_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    print(f"Start downloading {len(XHS_FEEDS)} notes' images...")
    
    for i, feed in enumerate(XHS_FEEDS):
        print(f"\n[{i+1}/{len(XHS_FEEDS)}] {feed['title']}")
        
        # Get note details
        detail = fetch_feed_detail(feed['id'], feed['xsecToken'])
        if not detail or 'data' not in detail:
            print(f"  Warning: Unable to get details")
            continue
        
        note = detail['data']['note']
        image_list = note.get('imageList', [])
        
        if not image_list:
            print(f"  Warning: No images")
            continue
        
        city = get_city_from_title(feed['title'])
        
        for j, img in enumerate(image_list):
            img_url = img.get('urlDefault', '')
            if not img_url:
                continue
            
            # Generate filename
            safe_title = "".join(c for c in feed['title'] if c.isalnum() or c in '._-')[:30]
            filename = f"{city}_{safe_title}_{j+1}.jpg"
            output_path = output_dir / filename
            
            print(f"  Downloading image {j+1}/{len(image_list)}: {filename}")
            if download_image(img_url, str(output_path)):
                downloaded.append({
                    "feed_id": feed['id'],
                    "title": feed['title'],
                    "city": city,
                    "image_path": str(output_path),
                    "image_url": img_url,
                    "width": img.get('width'),
                    "height": img.get('height')
                })
                print(f"    Success")
            else:
                print(f"    Failed")
    
    # Save download records
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            "downloaded_at": datetime.now().isoformat(),
            "total_feeds": len(XHS_FEEDS),
            "total_images": len(downloaded),
            "images": downloaded
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n Complete! Downloaded {len(downloaded)} images")
    print(f"Save location: {output_dir}")
    print(f"Manifest file: {manifest_path}")

if __name__ == "__main__":
    main()
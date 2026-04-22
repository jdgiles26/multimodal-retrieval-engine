#!/usr/bin/env python3
"""
下载小红书旅行自拍图片并扩充评测数据集
"""
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

# 小红书笔记数据（从 MCP 获取的）
XHS_FEEDS = [
    {"id": "69d3ceaf0000000021038158", "title": "分享日常", "xsecToken": "AB4XWq-F-Na-RYzxdV8lzjgHWomw3YlxhJ7C41p9k-8UE="},
    {"id": "69d089640000000021012a2f", "title": "樱花季的狐耳少女🌸春日限定自拍", "xsecToken": "ABvx8L4ej1EBSPbmEjxMPoBsqMwtiXhV-TvADzWWx_epc="},
    {"id": "69cd3fea000000001a023e96", "title": "🦊 晚安前的慵懒时光～", "xsecToken": "ABNlsriTzyIjiWTBo3NBJgzB2VLZfUHJZrrUGKlfHtNDI="},
    {"id": "69ca9878000000001a03099a", "title": "纽约自由女神像打卡", "xsecToken": "ABfvqUqPBg_pD-eM5pTkTTev6C7Il6TxqJI29GZCyq0bc="},
    {"id": "69c9cc38000000001a030837", "title": "纽约中央公园晨跑", "xsecToken": "ABwwYZaY3mL-eIteL7lcHrolNYSB9UYLjSKMdrUnbOZi4="},
    {"id": "69c7fb17000000002101111b", "title": "中央公园晨跑", "xsecToken": "AB7ACCHbT57KcWrpG2p6I-Uq_ydfnx-DZoFeVKjGZq4sg="},
    {"id": "69c6a3630000000022002b71", "title": "布鲁克林大桥", "xsecToken": "ABYRUyS3e2GJ5Sm4wRnh07mWTn2G5JK95mwIwHppafD-I="},
    {"id": "69c55453000000001a033d61", "title": "纽约时代广场", "xsecToken": "ABQG4s7df1JZTuRkCAWWmoGyf6xgTsfiIHcCDrcrdbKOY="},
    {"id": "69c3fc2400000000210380f0", "title": "狗拉雪橇", "xsecToken": "ABfRo1Q2C5rgq2rLEMbqaGSnkMvXAHqdBj7xqe7pk-gyA="},
    {"id": "69c28dfa000000001a023598", "title": "蓝湖温泉", "xsecToken": "ABidp9BhrbRON4qFrpj_uGcsvNUiijJ-6KSiC4S3yZVmw="},
    {"id": "69c1539f000000002103bcb2", "title": "冰岛·抵达", "xsecToken": "AB_kfURe4gsP127_wnLCE4fsPuXs_Qt7INwI-WROFl3y4="},
    {"id": "69bfeb71000000002200170d", "title": "凡尔赛宫·金碧辉煌", "xsecToken": "ABjq6k1V2xCB14iC2aYBLa7zLfx9DGQOB6vABicDnBLq4="},
    {"id": "69bea59800000000210111d5", "title": "巴黎·登顶", "xsecToken": "ABk36UgFfW4xtJhr6Igp9tlT4tU-KeWxTWYWmnqDxqvZg="},
    {"id": "69bd5a5a0000000022001ffb", "title": "巴黎·埃菲尔铁塔", "xsecToken": "ABlqDJS4khv37tfMvZTcY8Jul1JO9wDBATmKkQNlTFVhA="},
    {"id": "69bc0245000000001a021df7", "title": "V&A海滨·开普敦", "xsecToken": "ABEYYBYN5okyvfTSyHtkR-QWyYZAAfwt6Vb1-8Pn2fwfE="},
    {"id": "69bab925000000001a03161d", "title": "酒庄·南非红酒", "xsecToken": "ABahNm5OIFBtSM_c_c45j4lKKQ_lE6QT382v4BNhPlKn0="},
    {"id": "69b97093000000001a031eca", "title": "好望角·两大洋交汇点", "xsecToken": "ABY_rcpGryVoLGf2sBqOUsByCeoWK2G066BDTjVQZ-30w="},
    {"id": "69b8b018000000002102c971", "title": "桌山·开普敦", "xsecToken": "ABAMdPwxt6MTOzt70u2xMphT9zji9hBo06tL5LdFF8ckw="},
    {"id": "69b6ce8a000000001a0326f7", "title": "和袋鼠的夕阳合影🦘", "xsecToken": "ABG0jMWsnZShyptlpBQooXKGTx1M9uQc16e-5sIoy4G9w="},
    {"id": "69b56756000000001a02b0c5", "title": "蓝山一日游🏔️", "xsecToken": "ABZbDEL9WKGf3VCPgYek4yA1LKsyEszVLdG32nYuIPlfQ="},
    {"id": "69b431ad000000001a032af5", "title": "邦迪海滩的夏日时光🏖️", "xsecToken": "ABXpQZT5xaKb0_x096HGYbKvYBoqeVdyTU3lz1PPW-1is="},
    {"id": "69b2c87b000000001d01069b", "title": "悉尼歌剧院夜景自拍🦊", "xsecToken": "ABPgEiLUeNSR87GuuOvFgqOPPX8Sj36WpAp8wZBTPnW0w="},
    {"id": "69b18e7d000000001600aa12", "title": "悉尼歌剧院的夜晚✨", "xsecToken": "ABS7Bj9_rPbzSP-4Lp2rvlUqFMruRMmB3i4Azs3LtpF7w="},
    {"id": "69b03a77000000001b0173b4", "title": "东京深夜食堂🍜", "xsecToken": "ABkdXFbF6XDQFkTFZsbCCB4fDuwApegXE-99B9T__7aHA="},
    {"id": "69aecbb9000000001600bf09", "title": "东京最后一日｜台场高达与彩虹大桥", "xsecToken": "ABo87KvBkGPhsbxlk1t2NNNVOByDo3LykD_OwDZsNeBEY="},
    {"id": "69ad9159000000001d026f1e", "title": "涩谷夜景太绝了🗼", "xsecToken": "AB82awCNY0-qT07T7Cwg5CM8_-xzu5qV24Sg66T2kq-EI="},
    {"id": "69ac4cff0000000015038b0d", "title": "浅草寺祈福｜和服狐狸初体验👘🦊", "xsecToken": "ABFZeaObKpgJpBkvxMKep0VFpp1xtjHgZEmHus_VLNNFo="},
    {"id": "69abe20600000000150208d0", "title": "抵达东京｜歌舞伎町第一夜🌃🦊", "xsecToken": "ABPC7MFFc7tGmnzXSFLo7rBAxHLvx-VHWAswzECAvnJFU="},
    {"id": "69a95f1c00000000160082a2", "title": "🧳 收拾行李！明天出发东京", "xsecToken": "ABXpQZT5xaKb0_x096HGYbKvYBoqeVdyTU3lz1PPW-1is="},
]

# 城市映射
CITY_MAP = {
    "东京": "Tokyo",
    "悉尼": "Sydney",
    "开普敦": "Cape Town",
    "南非": "Cape Town",
    "巴黎": "Paris",
    "凡尔赛": "Paris",
    "冰岛": "Iceland",
    "蓝湖": "Iceland",
    "纽约": "New York",
    "中央公园": "New York",
    "时代广场": "New York",
    "布鲁克林": "New York",
    "自由女神": "New York",
    "袋鼠": "Sydney",
    "邦迪": "Sydney",
    "蓝山": "Sydney",
    "歌剧院": "Sydney",
    "桌山": "Cape Town",
    "好望角": "Cape Town",
    "酒庄": "Cape Town",
    "樱花": "Tokyo",
    "涩谷": "Tokyo",
    "浅草寺": "Tokyo",
    "歌舞伎": "Tokyo",
    "台场": "Tokyo",
}

def get_city_from_title(title):
    """从标题中提取城市"""
    for cn, en in CITY_MAP.items():
        if cn in title:
            return en
    return "Unknown"

def download_image(url, output_path):
    """下载图片"""
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
    """获取笔记详情"""
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
    output_dir = Path("/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval/data/xhs_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    
    print(f"开始下载 {len(XHS_FEEDS)} 条笔记的图片...")
    
    for i, feed in enumerate(XHS_FEEDS):
        print(f"\n[{i+1}/{len(XHS_FEEDS)}] {feed['title']}")
        
        # 获取笔记详情
        detail = fetch_feed_detail(feed['id'], feed['xsecToken'])
        if not detail or 'data' not in detail:
            print(f"  ⚠️ 无法获取详情")
            continue
        
        note = detail['data']['note']
        image_list = note.get('imageList', [])
        
        if not image_list:
            print(f"  ⚠️ 没有图片")
            continue
        
        city = get_city_from_title(feed['title'])
        
        for j, img in enumerate(image_list):
            img_url = img.get('urlDefault', '')
            if not img_url:
                continue
            
            # 生成文件名
            safe_title = "".join(c for c in feed['title'] if c.isalnum() or c in '._-')[:30]
            filename = f"{city}_{safe_title}_{j+1}.jpg"
            output_path = output_dir / filename
            
            print(f"  下载图片 {j+1}/{len(image_list)}: {filename}")
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
                print(f"    ✅ 成功")
            else:
                print(f"    ❌ 失败")
    
    # 保存下载记录
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            "downloaded_at": datetime.now().isoformat(),
            "total_feeds": len(XHS_FEEDS),
            "total_images": len(downloaded),
            "images": downloaded
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成！共下载 {len(downloaded)} 张图片")
    print(f"📁 保存位置: {output_dir}")
    print(f"📄 清单文件: {manifest_path}")

if __name__ == "__main__":
    main()
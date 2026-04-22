#!/usr/bin/env python3
"""
评测小红书旅行自拍数据集
支持多种多模态检索模型
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

# 数据集路径
DATASET_DIR = Path("/root/.openclaw/workspace/duckyo/projects/multimodal-retrieval/data/xhs_images")
DATASET_FILE = DATASET_DIR / "dataset.json"

class XHSDatasetEvaluator:
    """小红书数据集评测器"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or DATASET_FILE
        self.dataset = self._load_dataset()
        self.images = self.dataset.get("images", [])
        self.test_queries = self.dataset.get("test_queries", [])
        
    def _load_dataset(self) -> Dict:
        """加载数据集"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_image_paths(self) -> List[str]:
        """获取所有图片路径"""
        paths = []
        for img in self.images:
            path = DATASET_DIR / img["filename"]
            if path.exists():
                paths.append(str(path))
        return paths
    
    def get_image_by_id(self, image_id: str) -> Dict:
        """根据ID获取图片信息"""
        for img in self.images:
            if img["id"] == image_id:
                return img
        return None
    
    def evaluate_query(self, query: str, retrieved_ids: List[str]) -> Dict:
        """评测单个查询的结果"""
        # 找到对应的测试查询
        test_query = None
        for tq in self.test_queries:
            if tq["query"] == query:
                test_query = tq
                break
        
        if not test_query:
            return {"error": "Query not found in test set"}
        
        expected_ids = set(test_query["expected_ids"])
        retrieved_set = set(retrieved_ids)
        
        # 计算指标
        correct = len(expected_ids & retrieved_set)
        precision = correct / len(retrieved_ids) if retrieved_ids else 0
        recall = correct / len(expected_ids) if expected_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算 MRR (Mean Reciprocal Rank)
        mrr = 0
        for expected_id in expected_ids:
            if expected_id in retrieved_ids:
                rank = retrieved_ids.index(expected_id) + 1
                mrr += 1.0 / rank
        mrr /= len(expected_ids) if expected_ids else 1
        
        return {
            "query": query,
            "expected_ids": list(expected_ids),
            "retrieved_ids": retrieved_ids,
            "correct": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
            "hit@1": 1 if any(rid in expected_ids for rid in retrieved_ids[:1]) else 0,
            "hit@5": 1 if any(rid in expected_ids for rid in retrieved_ids[:5]) else 0,
        }
    
    def print_dataset_stats(self):
        """打印数据集统计信息"""
        print("=" * 60)
        print("📊 小红书旅行自拍数据集统计")
        print("=" * 60)
        print(f"数据集名称: {self.dataset.get('dataset_name')}")
        print(f"描述: {self.dataset.get('description')}")
        print(f"创建时间: {self.dataset.get('created_at')}")
        print(f"总图片数: {len(self.images)}")
        print(f"测试查询数: {len(self.test_queries)}")
        print(f"\n城市分布:")
        
        city_counts = {}
        for img in self.images:
            city = img.get("city", "Unknown")
            city_counts[city] = city_counts.get(city, 0) + 1
        
        for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
            print(f"  - {city}: {count}张")
        
        print(f"\n✅ 实际存在的图片: {len(self.get_image_paths())}张")
        print("=" * 60)
    
    def print_test_queries(self):
        """打印测试查询"""
        print("\n📝 测试查询列表:")
        print("-" * 60)
        for i, tq in enumerate(self.test_queries, 1):
            print(f"{i}. {tq['query']}")
            print(f"   期望结果: {', '.join(tq['expected_ids'])}")
            print()


def mock_retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    模拟检索函数 - 实际使用时替换为真实的模型检索
    这里只是根据关键词简单匹配
    """
    # 简单的关键词匹配
    keywords = {
        "悉尼": ["sydney_001", "sydney_002", "sydney_003", "sydney_004", "sydney_005"],
        "歌剧院": ["sydney_001", "sydney_002"],
        "埃菲尔": ["paris_001"],
        "巴黎": ["paris_001", "paris_002", "paris_003"],
        "冰岛": ["iceland_001", "iceland_002", "iceland_003"],
        "蓝湖": ["iceland_002"],
        "纽约": ["newyork_001", "newyork_002", "newyork_003", "newyork_004", "newyork_005"],
        "时代广场": ["newyork_001"],
        "东京": ["tokyo_001", "tokyo_002", "tokyo_003", "tokyo_004", "tokyo_005", "tokyo_006"],
        "浅草寺": ["tokyo_004"],
        "开普敦": ["capetown_001", "capetown_002", "capetown_003", "capetown_004"],
        "桌山": ["capetown_001"],
        "樱花": ["special_001"],
        "布鲁克林": ["newyork_002"],
        "好望角": ["capetown_002"],
        "涩谷": ["tokyo_001"],
    }
    
    results = []
    for keyword, ids in keywords.items():
        if keyword in query:
            results.extend(ids)
    
    return list(dict.fromkeys(results))[:top_k]  # 去重并限制数量


def main():
    """主函数"""
    evaluator = XHSDatasetEvaluator()
    
    # 打印数据集统计
    evaluator.print_dataset_stats()
    evaluator.print_test_queries()
    
    # 运行模拟评测
    print("\n🔍 运行模拟评测 (基于关键词匹配):")
    print("=" * 60)
    
    all_results = []
    for tq in evaluator.test_queries:
        query = tq["query"]
        retrieved = mock_retrieve(query, top_k=5)
        result = evaluator.evaluate_query(query, retrieved)
        all_results.append(result)
        
        print(f"\n查询: {query}")
        print(f"  检索结果: {retrieved}")
        print(f"  期望结果: {result['expected_ids']}")
        print(f"  Precision: {result['precision']:.2f}")
        print(f"  Recall: {result['recall']:.2f}")
        print(f"  F1: {result['f1']:.2f}")
        print(f"  MRR: {result['mrr']:.2f}")
        print(f"  Hit@1: {result['hit@1']}")
        print(f"  Hit@5: {result['hit@5']}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📈 评测汇总:")
    print("=" * 60)
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall"] for r in all_results) / len(all_results)
    avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
    avg_mrr = sum(r["mrr"] for r in all_results) / len(all_results)
    avg_hit1 = sum(r["hit@1"] for r in all_results) / len(all_results)
    avg_hit5 = sum(r["hit@5"] for r in all_results) / len(all_results)
    
    print(f"平均 Precision: {avg_precision:.3f}")
    print(f"平均 Recall: {avg_recall:.3f}")
    print(f"平均 F1: {avg_f1:.3f}")
    print(f"平均 MRR: {avg_mrr:.3f}")
    print(f"平均 Hit@1: {avg_hit1:.3f}")
    print(f"平均 Hit@5: {avg_hit5:.3f}")
    print("=" * 60)
    
    # 保存结果
    output_file = DATASET_DIR / "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "mock_keyword_matching",
            "num_queries": len(all_results),
            "metrics": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "mrr": avg_mrr,
                "hit@1": avg_hit1,
                "hit@5": avg_hit5,
            },
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评测结果已保存: {output_file}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Evaluate Xiaohongshu travel selfie dataset
Support multiple multimodal retrieval models
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time

# Dataset path
DATASET_DIR = Path(__file__).parent.parent / "data" / "xhs_images"
DATASET_FILE = DATASET_DIR / "dataset.json"

class XHSDatasetEvaluator:
    """Xiaohongshu dataset evaluator"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or DATASET_FILE
        self.dataset = self._load_dataset()
        self.images = self.dataset.get("images", [])
        self.test_queries = self.dataset.get("test_queries", [])
        
    def _load_dataset(self) -> Dict:
        """Load dataset"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_image_paths(self) -> List[str]:
        """Get all image paths"""
        paths = []
        for img in self.images:
            path = DATASET_DIR / img["filename"]
            if path.exists():
                paths.append(str(path))
        return paths
    
    def get_image_by_id(self, image_id: str) -> Dict:
        """Get image info by ID"""
        for img in self.images:
            if img["id"] == image_id:
                return img
        return None
    
    def evaluate_query(self, query: str, retrieved_ids: List[str]) -> Dict:
        """Evaluate single query result"""
        # Find corresponding test query
        test_query = None
        for tq in self.test_queries:
            if tq["query"] == query:
                test_query = tq
                break
        
        if not test_query:
            return {"error": "Query not found in test set"}
        
        expected_ids = set(test_query["expected_ids"])
        retrieved_set = set(retrieved_ids)
        
        # Calculate metrics
        correct = len(expected_ids & retrieved_set)
        precision = correct / len(retrieved_ids) if retrieved_ids else 0
        recall = correct / len(expected_ids) if expected_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate MRR (Mean Reciprocal Rank)
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
        """Print dataset statistics"""
        print("=" * 60)
        print(" Xiaohongshu travel selfie dataset statistics")
        print("=" * 60)
        print(f"Dataset name: {self.dataset.get('dataset_name')}")
        print(f"Description: {self.dataset.get('description')}")
        print(f"Created at: {self.dataset.get('created_at')}")
        print(f"Total images: {len(self.images)}")
        print(f"Test queries: {len(self.test_queries)}")
        print(f"\nCity distribution:")
        
        city_counts = {}
        for img in self.images:
            city = img.get("city", "Unknown")
            city_counts[city] = city_counts.get(city, 0) + 1
        
        for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
            print(f"  - {city}: {count} images")
        
        print(f"\n Actually existing images: {len(self.get_image_paths())} images")
        print("=" * 60)
    
    def print_test_queries(self):
        """Print test queries"""
        print("\n Test query list:")
        print("-" * 60)
        for i, tq in enumerate(self.test_queries, 1):
            print(f"{i}. {tq['query']}")
            print(f"   Expected results: {', '.join(tq['expected_ids'])}")
            print()


def mock_retrieve(query: str, top_k: int = 5) -> List[str]:
    """
    Mock retrieval function - replace with actual model retrieval in production
    This is just simple keyword matching
    """
    # Simple keyword matching
    keywords = {
        "Sydney": ["sydney_001", "sydney_002", "sydney_003", "sydney_004", "sydney_005"],
        "Opera House": ["sydney_001", "sydney_002"],
        "Eiffel": ["paris_001"],
        "Paris": ["paris_001", "paris_002", "paris_003"],
        "Iceland": ["iceland_001", "iceland_002", "iceland_003"],
        "Blue Lagoon": ["iceland_002"],
        "New York": ["newyork_001", "newyork_002", "newyork_003", "newyork_004", "newyork_005"],
        "Times Square": ["newyork_001"],
        "Tokyo": ["tokyo_001", "tokyo_002", "tokyo_003", "tokyo_004", "tokyo_005", "tokyo_006"],
        "Sensoji Temple": ["tokyo_004"],
        "Cape Town": ["capetown_001", "capetown_002", "capetown_003", "capetown_004"],
        "Table Mountain": ["capetown_001"],
        "cherry blossom": ["special_001"],
        "Brooklyn": ["newyork_002"],
        "Cape of Good Hope": ["capetown_002"],
        "Shibuya": ["tokyo_001"],
    }
    
    results = []
    for keyword, ids in keywords.items():
        if keyword in query:
            results.extend(ids)
    
    return list(dict.fromkeys(results))[:top_k]  # Deduplicate and limit number


def main():
    """Main function"""
    evaluator = XHSDatasetEvaluator()
    
    # Print dataset statistics
    evaluator.print_dataset_stats()
    evaluator.print_test_queries()
    
    # Run mock evaluation
    print("\n Run mock evaluation (based on keyword matching):")
    print("=" * 60)
    
    all_results = []
    for tq in evaluator.test_queries:
        query = tq["query"]
        retrieved = mock_retrieve(query, top_k=5)
        result = evaluator.evaluate_query(query, retrieved)
        all_results.append(result)
        
        print(f"\nQuery: {query}")
        print(f"  Retrieval results: {retrieved}")
        print(f"  Expected results: {result['expected_ids']}")
        print(f"  Precision: {result['precision']:.2f}")
        print(f"  Recall: {result['recall']:.2f}")
        print(f"  F1: {result['f1']:.2f}")
        print(f"  MRR: {result['mrr']:.2f}")
        print(f"  Hit@1: {result['hit@1']}")
        print(f"  Hit@5: {result['hit@5']}")
    
    # Summary results
    print("\n" + "=" * 60)
    print(" Evaluation summary:")
    print("=" * 60)
    avg_precision = sum(r["precision"] for r in all_results) / len(all_results)
    avg_recall = sum(r["recall"] for r in all_results) / len(all_results)
    avg_f1 = sum(r["f1"] for r in all_results) / len(all_results)
    avg_mrr = sum(r["mrr"] for r in all_results) / len(all_results)
    avg_hit1 = sum(r["hit@1"] for r in all_results) / len(all_results)
    avg_hit5 = sum(r["hit@5"] for r in all_results) / len(all_results)
    
    print(f"Average Precision: {avg_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Average F1: {avg_f1:.3f}")
    print(f"Average MRR: {avg_mrr:.3f}")
    print(f"Average Hit@1: {avg_hit1:.3f}")
    print(f"Average Hit@5: {avg_hit5:.3f}")
    print("=" * 60)
    
    # Save results
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
    
    print(f"\n Evaluation results saved: {output_file}")


if __name__ == "__main__":
    main()
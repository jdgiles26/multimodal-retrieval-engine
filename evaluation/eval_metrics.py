"""
Evaluation metrics calculation module
Implement standard evaluation metrics for multimodal retrieval engine
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    # Accuracy metrics
    recall_at_k: Dict[int, float]  # Recall@K
    precision_at_k: Dict[int, float]  # Precision@K
    ndcg_at_k: Dict[int, float]  # NDCG@K
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    
    # Efficiency metrics
    latency_ms: float  # Latency (milliseconds)
    throughput_qps: Optional[float] = None  # Throughput (QPS)
    
    # Multimodal-specific metrics
    cross_modal_accuracy: Optional[float] = None  # Cross-modal retrieval accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "map": self.map_score,
            "latency_ms": self.latency_ms,
            "throughput_qps": self.throughput_qps,
            "cross_modal_accuracy": self.cross_modal_accuracy,
        }
    
    def __str__(self) -> str:
        lines = [
            " Retrieval evaluation metrics",
            "=" * 40,
            f"Recall@K: {self.recall_at_k}",
            f"Precision@K: {self.precision_at_k}",
            f"NDCG@K: {self.ndcg_at_k}",
            f"MRR: {self.mrr:.4f}",
            f"MAP: {self.map_score:.4f}",
            f"Latency: {self.latency_ms:.2f}ms",
        ]
        if self.throughput_qps:
            lines.append(f"Throughput: {self.throughput_qps:.2f} QPS")
        if self.cross_modal_accuracy:
            lines.append(f"Cross-Modal Acc: {self.cross_modal_accuracy:.4f}")
        return "\n".join(lines)


class MetricsCalculator:
    """Evaluation metrics calculator"""
    
    @staticmethod
    def calculate_recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, float]:
        """
        Calculate Recall@K
        Recall@K = (number of relevant documents in top K) / (total number of relevant documents)
        """
        if not relevant_ids:
            return {k: 0.0 for k in k_values}
        
        results = {}
        for k in k_values:
            top_k = retrieved_ids[:k]
            relevant_in_top_k = len(set(top_k) & set(relevant_ids))
            results[k] = relevant_in_top_k / len(relevant_ids)
        return results
    
    @staticmethod
    def calculate_precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, float]:
        """
        Calculate Precision@K
        Precision@K = (number of relevant documents in top K) / K
        """
        if not retrieved_ids:
            return {k: 0.0 for k in k_values}
        
        results = {}
        for k in k_values:
            top_k = retrieved_ids[:k]
            relevant_in_top_k = len(set(top_k) & set(relevant_ids))
            results[k] = relevant_in_top_k / min(k, len(retrieved_ids))
        return results
    
    @staticmethod
    def calculate_dcg_at_k(
        retrieved_ids: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Calculate DCG@K (Discounted Cumulative Gain)
        DCG = sum(rel_i / log2(i + 1))
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)
        return dcg
    
    @staticmethod
    def calculate_ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: Dict[str, float],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[int, float]:
        """
        Calculate NDCG@K (Normalized DCG)
        NDCG = DCG / IDCG (Ideal DCG)
        """
        results = {}
        
        # Calculate ideal ordering (sorted by relevance score descending)
        ideal_ordering = sorted(
            relevant_ids,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )
        
        for k in k_values:
            dcg = MetricsCalculator.calculate_dcg_at_k(
                retrieved_ids, relevance_scores, k
            )
            idcg = MetricsCalculator.calculate_dcg_at_k(
                ideal_ordering, relevance_scores, k
            )
            
            if idcg > 0:
                results[k] = dcg / idcg
            else:
                results[k] = 0.0
        
        return results
    
    @staticmethod
    def calculate_mrr(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank)
        MRR = 1 / rank_of_first_relevant
        """
        if not relevant_ids:
            return 0.0
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
    
    @staticmethod
    def calculate_map(
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate MAP (Mean Average Precision)
        AP = average of precision@k for each relevant document
        """
        if not relevant_ids:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for k, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precisions.append(precision_at_k)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_ids)
    
    @staticmethod
    def calculate_all_metrics(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        relevance_scores: Dict[str, float],
        latency_ms: float,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> RetrievalMetrics:
        """Calculate all metrics"""
        return RetrievalMetrics(
            recall_at_k=MetricsCalculator.calculate_recall_at_k(
                retrieved_ids, relevant_ids, k_values
            ),
            precision_at_k=MetricsCalculator.calculate_precision_at_k(
                retrieved_ids, relevant_ids, k_values
            ),
            ndcg_at_k=MetricsCalculator.calculate_ndcg_at_k(
                retrieved_ids, relevant_ids, relevance_scores, k_values
            ),
            mrr=MetricsCalculator.calculate_mrr(retrieved_ids, relevant_ids),
            map_score=MetricsCalculator.calculate_map(retrieved_ids, relevant_ids),
            latency_ms=latency_ms,
        )


class AggregatedMetrics:
    """Aggregate evaluation results from multiple queries"""
    
    def __init__(self):
        self.query_metrics: List[RetrievalMetrics] = []
        self.query_types: Dict[str, List[RetrievalMetrics]] = {
            "text": [],
            "image": [],
            "hybrid": [],
        }
        self.difficulty_levels: Dict[str, List[RetrievalMetrics]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
    
    def add_query_result(
        self,
        metrics: RetrievalMetrics,
        query_type: str,
        difficulty: str
    ):
        """Add evaluation result for a single query"""
        self.query_metrics.append(metrics)
        
        if query_type in self.query_types:
            self.query_types[query_type].append(metrics)
        
        if difficulty in self.difficulty_levels:
            self.difficulty_levels[difficulty].append(metrics)
    
    def _aggregate_metrics(
        self,
        metrics_list: List[RetrievalMetrics]
    ) -> Dict[str, Any]:
        """Aggregate metrics list"""
        if not metrics_list:
            return {}
        
        # Aggregate Recall@K
        recall_at_k = {}
        for k in [1, 3, 5, 10]:
            values = [m.recall_at_k.get(k, 0.0) for m in metrics_list]
            recall_at_k[k] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        # Aggregate Precision@K
        precision_at_k = {}
        for k in [1, 3, 5, 10]:
            values = [m.precision_at_k.get(k, 0.0) for m in metrics_list]
            precision_at_k[k] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        # Aggregate NDCG@K
        ndcg_at_k = {}
        for k in [1, 3, 5, 10]:
            values = [m.ndcg_at_k.get(k, 0.0) for m in metrics_list]
            ndcg_at_k[k] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        # Aggregate other metrics
        mrr_values = [m.mrr for m in metrics_list]
        map_values = [m.map_score for m in metrics_list]
        latency_values = [m.latency_ms for m in metrics_list]
        
        return {
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "ndcg_at_k": ndcg_at_k,
            "mrr": {
                "mean": np.mean(mrr_values),
                "std": np.std(mrr_values),
            },
            "map": {
                "mean": np.mean(map_values),
                "std": np.std(map_values),
            },
            "latency_ms": {
                "mean": np.mean(latency_values),
                "std": np.std(latency_values),
                "p50": np.percentile(latency_values, 50),
                "p95": np.percentile(latency_values, 95),
                "p99": np.percentile(latency_values, 99),
            },
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary"""
        summary = {
            "overall": self._aggregate_metrics(self.query_metrics),
            "by_query_type": {},
            "by_difficulty": {},
        }
        
        for query_type, metrics in self.query_types.items():
            if metrics:
                summary["by_query_type"][query_type] = self._aggregate_metrics(metrics)
        
        for difficulty, metrics in self.difficulty_levels.items():
            if metrics:
                summary["by_difficulty"][difficulty] = self._aggregate_metrics(metrics)
        
        return summary
    
    def print_summary(self):
        """Print evaluation summary"""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print(" Duckyo Multimodal Retrieval Engine Evaluation Results")
        print("=" * 60)
        
        # Overall metrics
        overall = summary["overall"]
        print("\n Overall metrics")
        print("-" * 40)
        self._print_metrics(overall)
        
        # By query type
        print("\n By query type")
        print("-" * 40)
        for query_type, metrics in summary["by_query_type"].items():
            print(f"\n  [{query_type.upper()}]")
            self._print_metrics(metrics, indent=4)
        
        # By difficulty
        print("\n By difficulty level")
        print("-" * 40)
        for difficulty, metrics in summary["by_difficulty"].items():
            print(f"\n  [{difficulty.upper()}]")
            self._print_metrics(metrics, indent=4)
    
    def _print_metrics(self, metrics: Dict, indent: int = 0):
        """Print metrics"""
        prefix = " " * indent
        
        print(f"{prefix}Recall@5: {metrics['recall_at_k'][5]['mean']:.4f}")
        print(f"{prefix}Precision@5: {metrics['precision_at_k'][5]['mean']:.4f}")
        print(f"{prefix}NDCG@5: {metrics['ndcg_at_k'][5]['mean']:.4f}")
        print(f"{prefix}MRR: {metrics['mrr']['mean']:.4f}")
        print(f"{prefix}MAP: {metrics['map']['mean']:.4f}")
        print(f"{prefix}Latency (p50/p95): {metrics['latency_ms']['p50']:.1f}ms / {metrics['latency_ms']['p95']:.1f}ms")

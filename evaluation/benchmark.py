"""
Benchmark Script
Quickly run benchmarks to compare performance across different configurations
"""
import asyncio
import time
import json
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Benchmark Result"""
    name: str
    config: Dict
    avg_latency_ms: float
    throughput_qps: float
    recall_at_5: float
    ndcg_at_5: float
    mrr: float
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "config": self.config,
            "avg_latency_ms": self.avg_latency_ms,
            "throughput_qps": self.throughput_qps,
            "recall_at_5": self.recall_at_5,
            "ndcg_at_5": self.ndcg_at_5,
            "mrr": self.mrr,
        }


class BenchmarkRunner:
    """
    Benchmark Runner
    Compare performance across different configurations
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_config(
        self,
        name: str,
        config: Dict,
        test_queries: List[str],
        iterations: int = 5
    ) -> BenchmarkResult:
        """
        Test a single configuration
        """
        print(f"\n Testing config: {name}")
        print(f"   Config: {config}")
        
        latencies = []
        all_results = []
        
        for i in range(iterations):
            for query in test_queries:
                start = time.time()
                
                # Execute search according to config
                if config.get("search_type") == "text":
                    results = await self.retriever.search(
                        query_text=query,
                        top_k=config.get("top_k", 10),
                        use_rerank=config.get("use_rerank", True),
                    )
                else:
                    results = await self.retriever.search(
                        query_text=query,
                        top_k=config.get("top_k", 10),
                        use_rerank=config.get("use_rerank", True),
                    )
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                all_results.append(results)
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        total_time = sum(latencies) / 1000  # Convert to seconds
        throughput = len(test_queries) * iterations / total_time if total_time > 0 else 0
        
        # Simplified metrics (should use evaluation dataset in practice)
        result = BenchmarkResult(
            name=name,
            config=config,
            avg_latency_ms=avg_latency,
            throughput_qps=throughput,
            recall_at_5=0.0,  # Need real labels
            ndcg_at_5=0.0,
            mrr=0.0,
        )
        
        print(f"    Average latency: {avg_latency:.2f}ms")
        print(f"    Throughput: {throughput:.2f} QPS")
        
        return result
    
    async def run_benchmarks(self):
        """Run all benchmarks"""
        test_queries = [
            "Tokyo Tower night view",
            "Sydney Opera House",
            "Hanfu travel",
            "Beach vacation",
            "Eiffel Tower",
        ]
        
        configs = [
            {
                "name": "baseline_rerank",
                "config": {
                    "search_type": "text",
                    "top_k": 10,
                    "use_rerank": True,
                },
            },
            {
                "name": "baseline_no_rerank",
                "config": {
                    "search_type": "text",
                    "top_k": 10,
                    "use_rerank": False,
                },
            },
            {
                "name": "top_5_rerank",
                "config": {
                    "search_type": "text",
                    "top_k": 5,
                    "use_rerank": True,
                },
            },
            {
                "name": "top_20_rerank",
                "config": {
                    "search_type": "text",
                    "top_k": 20,
                    "use_rerank": True,
                },
            },
        ]
        
        for cfg in configs:
            result = await self.benchmark_config(
                name=cfg["name"],
                config=cfg["config"],
                test_queries=test_queries,
                iterations=3,
            )
            self.results.append(result)
    
    def generate_report(self) -> str:
        """Generate benchmark report"""
        lines = [
            "# Duckyo Multimodal Retrieval Engine Benchmark Report",
            "",
            "## Performance Comparison",
            "",
            "| Config | Avg Latency (ms) | Throughput (QPS) |",
            "|------|------------------|------------------|",
        ]
        
        for r in self.results:
            lines.append(f"| {r.name} | {r.avg_latency_ms:.2f} | {r.throughput_qps:.2f} |")
        
        lines.extend([
            "",
            "## Detailed Results",
            "",
            "```json",
            json.dumps([r.to_dict() for r in self.results], indent=2),
            "```",
        ])
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str = "evaluation/results/benchmark.json"):
        """Save results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"\n Benchmark results saved: {output_path}")


async def main():
    """Main function"""
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    
    retriever = LlamaIndexMultimodalRetriever()
    runner = BenchmarkRunner(retriever)
    
    await runner.run_benchmarks()
    
    # Save results
    runner.save_results()
    
    # Generate and save report
    report = runner.generate_report()
    report_path = Path("evaluation/results/benchmark_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f" Report saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

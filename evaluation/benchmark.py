"""
基准测试脚本
快速运行基准测试，对比不同配置的性能
"""
import asyncio
import time
import json
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """基准测试结果"""
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
    基准测试运行器
    对比不同配置的性能表现
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
        测试单个配置
        """
        print(f"\n🔧 测试配置: {name}")
        print(f"   配置: {config}")
        
        latencies = []
        all_results = []
        
        for i in range(iterations):
            for query in test_queries:
                start = time.time()
                
                # 根据配置执行搜索
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
        
        # 计算指标
        avg_latency = sum(latencies) / len(latencies)
        total_time = sum(latencies) / 1000  # 转换为秒
        throughput = len(test_queries) * iterations / total_time if total_time > 0 else 0
        
        # 简化指标（实际应该使用评测数据集）
        result = BenchmarkResult(
            name=name,
            config=config,
            avg_latency_ms=avg_latency,
            throughput_qps=throughput,
            recall_at_5=0.0,  # 需要真实标签
            ndcg_at_5=0.0,
            mrr=0.0,
        )
        
        print(f"   ✅ 平均延迟: {avg_latency:.2f}ms")
        print(f"   ✅ 吞吐量: {throughput:.2f} QPS")
        
        return result
    
    async def run_benchmarks(self):
        """运行所有基准测试"""
        test_queries = [
            "东京塔夜景",
            "悉尼歌剧院",
            "汉服旅行",
            "海滩度假",
            "埃菲尔铁塔",
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
        """生成基准测试报告"""
        lines = [
            "# Duckyo 多模态检索引擎基准测试报告",
            "",
            "## 性能对比",
            "",
            "| 配置 | 平均延迟 (ms) | 吞吐量 (QPS) |",
            "|------|--------------|-------------|",
        ]
        
        for r in self.results:
            lines.append(f"| {r.name} | {r.avg_latency_ms:.2f} | {r.throughput_qps:.2f} |")
        
        lines.extend([
            "",
            "## 详细结果",
            "",
            "```json",
            json.dumps([r.to_dict() for r in self.results], indent=2),
            "```",
        ])
        
        return "\n".join(lines)
    
    def save_results(self, output_path: str = "evaluation/results/benchmark.json"):
        """保存结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        print(f"\n✅ 基准测试结果已保存: {output_path}")


async def main():
    """主函数"""
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    
    retriever = LlamaIndexMultimodalRetriever()
    runner = BenchmarkRunner(retriever)
    
    await runner.run_benchmarks()
    
    # 保存结果
    runner.save_results()
    
    # 生成并保存报告
    report = runner.generate_report()
    report_path = Path("evaluation/results/benchmark_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 报告已保存: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())

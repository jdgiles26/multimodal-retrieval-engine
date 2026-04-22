"""
评测执行器
集成 LlamaIndex 检索引擎和 Langfuse 进行端到端评测
"""
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

# 导入评测相关模块
from evaluation.eval_dataset import EvalDataset, QueryRelevance
from evaluation.eval_metrics import MetricsCalculator, AggregatedMetrics, RetrievalMetrics

# 导入检索引擎
try:
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LlamaIndexMultimodalRetriever = None
    LLAMAINDEX_AVAILABLE = False
    logger.warning("LlamaIndex engine not available")

# Langfuse 集成
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not available")


class EvalRunner:
    """
    评测执行器
    运行评测数据集并计算指标
    """
    
    def __init__(
        self,
        retriever: Optional[LlamaIndexMultimodalRetriever] = None,
        dataset_path: str = "evaluation/eval_dataset.json",
        enable_langfuse: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.retriever = retriever
        self.enable_langfuse = enable_langfuse and LANGFUSE_AVAILABLE
        
        # 加载评测数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dataset = EvalDataset.from_dict(data)
        
        # 初始化 Langfuse
        self.langfuse = None
        if self.enable_langfuse:
            try:
                from config.settings import settings
                if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
                    self.langfuse = Langfuse(
                        public_key=settings.LANGFUSE_PUBLIC_KEY,
                        secret_key=settings.LANGFUSE_SECRET_KEY,
                        host=settings.LANGFUSE_HOST,
                    )
                    logger.info("Langfuse initialized for evaluation")
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")
        
        # 评测结果
        self.aggregated_metrics = AggregatedMetrics()
        self.detailed_results: List[Dict] = []
    
    async def run_single_query(
        self,
        query: QueryRelevance,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行单个查询评测
        """
        start_time = time.time()
        
        # 构建 Langfuse trace
        trace = None
        if self.langfuse:
            trace = self.langfuse.trace(
                name="eval_query",
                id=experiment_id,
                input={
                    "query": query.query,
                    "query_type": query.query_type,
                    "difficulty": query.difficulty,
                },
                metadata={
                    "category": query.category,
                    "expected_ids": query.relevant_doc_ids,
                }
            )
        
        # 执行检索
        try:
            if query.query_type == "text":
                results = await self.retriever.search(
                    query_text=query.query,
                    top_k=10,
                    use_rerank=True,
                )
            elif query.query_type == "image":
                results = await self.retriever.search(
                    query_image=query.query_image,
                    top_k=10,
                    use_rerank=True,
                )
            elif query.query_type == "hybrid":
                results = await self.retriever.search(
                    query_text=query.query,
                    query_image=query.query_image,
                    top_k=10,
                    use_rerank=True,
                )
            else:
                raise ValueError(f"Unknown query type: {query.query_type}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 提取检索到的文档ID
            retrieved_ids = [r["id"] for r in results]
            
            # 计算指标
            metrics = MetricsCalculator.calculate_all_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=query.relevant_doc_ids,
                relevance_scores=query.relevance_scores,
                latency_ms=latency_ms,
            )
            
            # 记录到 Langfuse
            if trace:
                trace.score(name="recall@5", value=metrics.recall_at_k.get(5, 0.0))
                trace.score(name="ndcg@5", value=metrics.ndcg_at_k.get(5, 0.0))
                trace.score(name="mrr", value=metrics.mrr)
                trace.score(name="latency_ms", value=latency_ms)
                trace.end(output={
                    "retrieved_ids": retrieved_ids[:5],
                    "results_count": len(results),
                })
            
            # 构建详细结果
            result_detail = {
                "query": query.query,
                "query_type": query.query_type,
                "difficulty": query.difficulty,
                "category": query.category,
                "expected_ids": query.relevant_doc_ids,
                "retrieved_ids": retrieved_ids,
                "metrics": metrics.to_dict(),
                "latency_ms": latency_ms,
                "success": True,
            }
            
            # 添加到聚合器
            self.aggregated_metrics.add_query_result(
                metrics, query.query_type, query.difficulty
            )
            
            return result_detail
            
        except Exception as e:
            logger.error(f"Query evaluation failed: {e}")
            latency_ms = (time.time() - start_time) * 1000
            
            if trace:
                trace.score(name="success", value=0.0)
                trace.end(output={"error": str(e)})
            
            return {
                "query": query.query,
                "query_type": query.query_type,
                "difficulty": query.difficulty,
                "category": query.category,
                "expected_ids": query.relevant_doc_ids,
                "retrieved_ids": [],
                "metrics": {},
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e),
            }
    
    async def run_evaluation(
        self,
        experiment_name: str = "duckyo_eval",
        max_queries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行完整评测
        """
        logger.info(f"Starting evaluation: {experiment_name}")
        logger.info(f"Dataset: {self.dataset.name}")
        logger.info(f"Total queries: {len(self.dataset.queries)}")
        
        # 创建 Langfuse dataset run
        dataset_run = None
        if self.langfuse:
            try:
                # 创建或获取 dataset
                dataset = self.langfuse.create_dataset(
                    name=self.dataset.name,
                    description=self.dataset.description,
                )
                logger.info(f"Langfuse dataset created: {dataset.id}")
            except Exception as e:
                logger.warning(f"Failed to create Langfuse dataset: {e}")
        
        # 选择查询子集
        queries = self.dataset.queries
        if max_queries:
            queries = queries[:max_queries]
        
        # 执行所有查询
        for i, query in enumerate(queries):
            logger.info(f"Evaluating query {i+1}/{len(queries)}: {query.query[:50]}...")
            
            result = await self.run_single_query(
                query,
                experiment_id=f"{experiment_name}_{i}"
            )
            self.detailed_results.append(result)
        
        # 获取汇总结果
        summary = self.aggregated_metrics.get_summary()
        
        # 保存评测结果
        output = {
            "experiment_name": experiment_name,
            "dataset": {
                "name": self.dataset.name,
                "description": self.dataset.description,
                "corpus_size": self.dataset.corpus_size,
                "query_count": len(queries),
            },
            "summary": summary,
            "detailed_results": self.detailed_results,
        }
        
        # 保存到文件
        output_path = Path(f"evaluation/results/{experiment_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved: {output_path}")
        
        # 打印汇总
        self.aggregated_metrics.print_summary()
        
        return output
    
    def generate_report(self, experiment_name: str) -> str:
        """生成评测报告"""
        summary = self.aggregated_metrics.get_summary()
        overall = summary["overall"]
        
        report_lines = [
            "# Duckyo 多模态检索引擎评测报告",
            "",
            f"**实验名称**: {experiment_name}",
            f"**数据集**: {self.dataset.name}",
            f"**语料库大小**: {self.dataset.corpus_size}",
            f"**查询数量**: {len(self.detailed_results)}",
            "",
            "## 总体指标",
            "",
            "| 指标 | 均值 | 标准差 |",
            "|------|------|--------|",
            f"| Recall@5 | {overall['recall_at_k'][5]['mean']:.4f} | {overall['recall_at_k'][5]['std']:.4f} |",
            f"| Precision@5 | {overall['precision_at_k'][5]['mean']:.4f} | {overall['precision_at_k'][5]['std']:.4f} |",
            f"| NDCG@5 | {overall['ndcg_at_k'][5]['mean']:.4f} | {overall['ndcg_at_k'][5]['std']:.4f} |",
            f"| MRR | {overall['mrr']['mean']:.4f} | {overall['mrr']['std']:.4f} |",
            f"| MAP | {overall['map']['mean']:.4f} | {overall['map']['std']:.4f} |",
            f"| Latency (p50) | {overall['latency_ms']['p50']:.1f}ms | - |",
            f"| Latency (p95) | {overall['latency_ms']['p95']:.1f}ms | - |",
            "",
            "## 按查询类型",
            "",
        ]
        
        for query_type, metrics in summary.get("by_query_type", {}).items():
            report_lines.extend([
                f"### {query_type.upper()}",
                "",
                f"- Recall@5: {metrics['recall_at_k'][5]['mean']:.4f}",
                f"- NDCG@5: {metrics['ndcg_at_k'][5]['mean']:.4f}",
                f"- MRR: {metrics['mrr']['mean']:.4f}",
                "",
            ])
        
        report_lines.extend([
            "## 按难度级别",
            "",
        ])
        
        for difficulty, metrics in summary.get("by_difficulty", {}).items():
            report_lines.extend([
                f"### {difficulty.upper()}",
                "",
                f"- Recall@5: {metrics['recall_at_k'][5]['mean']:.4f}",
                f"- NDCG@5: {metrics['ndcg_at_k'][5]['mean']:.4f}",
                f"- MRR: {metrics['mrr']['mean']:.4f}",
                "",
            ])
        
        return "\n".join(report_lines)


async def main():
    """主函数 - 运行评测"""
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    
    # 初始化检索引擎
    retriever = LlamaIndexMultimodalRetriever()
    
    # 创建评测执行器
    runner = EvalRunner(
        retriever=retriever,
        dataset_path="evaluation/eval_dataset.json",
        enable_langfuse=True,
    )
    
    # 运行评测
    results = await runner.run_evaluation(
        experiment_name="duckyo_baseline_v1",
        max_queries=None,  # 运行所有查询
    )
    
    # 生成报告
    report = runner.generate_report("duckyo_baseline_v1")
    
    # 保存报告
    report_path = Path("evaluation/results/duckyo_baseline_v1_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 评测完成！")
    print(f"   结果文件: evaluation/results/duckyo_baseline_v1.json")
    print(f"   报告文件: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
"""
Evaluation runner
Integrate LlamaIndex retriever and Langfuse for end-to-end evaluation
"""
import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

# Import evaluation modules
from evaluation.eval_dataset import EvalDataset, QueryRelevance
from evaluation.eval_metrics import MetricsCalculator, AggregatedMetrics, RetrievalMetrics

# Import retriever engine
try:
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LlamaIndexMultimodalRetriever = None
    LLAMAINDEX_AVAILABLE = False
    logger.warning("LlamaIndex engine not available")

# Langfuse integration
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.warning("Langfuse not available")


class EvalRunner:
    """
    Evaluation runner
    Run evaluation dataset and calculate metrics
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
        
        # Load evaluation dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.dataset = EvalDataset.from_dict(data)
        
        # Initialize Langfuse
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
        
        # Evaluation results
        self.aggregated_metrics = AggregatedMetrics()
        self.detailed_results: List[Dict] = []
    
    async def run_single_query(
        self,
        query: QueryRelevance,
        experiment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute single query evaluation
        """
        start_time = time.time()
        
        # Build Langfuse trace
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
        
        # Execute retrieval
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
            
            # Extract retrieved document IDs
            retrieved_ids = [r["id"] for r in results]
            
            # Calculate metrics
            metrics = MetricsCalculator.calculate_all_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=query.relevant_doc_ids,
                relevance_scores=query.relevance_scores,
                latency_ms=latency_ms,
            )
            
            # Record to Langfuse
            if trace:
                trace.score(name="recall@5", value=metrics.recall_at_k.get(5, 0.0))
                trace.score(name="ndcg@5", value=metrics.ndcg_at_k.get(5, 0.0))
                trace.score(name="mrr", value=metrics.mrr)
                trace.score(name="latency_ms", value=latency_ms)
                trace.end(output={
                    "retrieved_ids": retrieved_ids[:5],
                    "results_count": len(results),
                })
            
            # Build detailed results
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
            
            # Add to aggregator
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
        Run full evaluation
        """
        logger.info(f"Starting evaluation: {experiment_name}")
        logger.info(f"Dataset: {self.dataset.name}")
        logger.info(f"Total queries: {len(self.dataset.queries)}")
        
        # Create Langfuse dataset run
        dataset_run = None
        if self.langfuse:
            try:
                # Create or get dataset
                dataset = self.langfuse.create_dataset(
                    name=self.dataset.name,
                    description=self.dataset.description,
                )
                logger.info(f"Langfuse dataset created: {dataset.id}")
            except Exception as e:
                logger.warning(f"Failed to create Langfuse dataset: {e}")
        
        # Select query subset
        queries = self.dataset.queries
        if max_queries:
            queries = queries[:max_queries]
        
        # Execute all queries
        for i, query in enumerate(queries):
            logger.info(f"Evaluating query {i+1}/{len(queries)}: {query.query[:50]}...")
            
            result = await self.run_single_query(
                query,
                experiment_id=f"{experiment_name}_{i}"
            )
            self.detailed_results.append(result)
        
        # Get summary results
        summary = self.aggregated_metrics.get_summary()
        
        # Save evaluation results
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
        
        # Save to file
        output_path = Path(f"evaluation/results/{experiment_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved: {output_path}")
        
        # Print summary
        self.aggregated_metrics.print_summary()
        
        return output
    
    def generate_report(self, experiment_name: str) -> str:
        """Generate evaluation report"""
        summary = self.aggregated_metrics.get_summary()
        overall = summary["overall"]
        
        report_lines = [
            "# Duckyo Multimodal Retrieval Engine Evaluation Report",
            "",
            f"**Experiment name**: {experiment_name}",
            f"**Dataset**: {self.dataset.name}",
            f"**Corpus size**: {self.dataset.corpus_size}",
            f"**Query count**: {len(self.detailed_results)}",
            "",
            "## Overall metrics",
            "",
            "| Metric | Mean | Std |",
            "|------|------|--------|",
            f"| Recall@5 | {overall['recall_at_k'][5]['mean']:.4f} | {overall['recall_at_k'][5]['std']:.4f} |",
            f"| Precision@5 | {overall['precision_at_k'][5]['mean']:.4f} | {overall['precision_at_k'][5]['std']:.4f} |",
            f"| NDCG@5 | {overall['ndcg_at_k'][5]['mean']:.4f} | {overall['ndcg_at_k'][5]['std']:.4f} |",
            f"| MRR | {overall['mrr']['mean']:.4f} | {overall['mrr']['std']:.4f} |",
            f"| MAP | {overall['map']['mean']:.4f} | {overall['map']['std']:.4f} |",
            f"| Latency (p50) | {overall['latency_ms']['p50']:.1f}ms | - |",
            f"| Latency (p95) | {overall['latency_ms']['p95']:.1f}ms | - |",
            "",
            "## By query type",
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
            "## By difficulty level",
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
    """Main function - run evaluation"""
    from core.llamaindex_engine import LlamaIndexMultimodalRetriever
    
    # Initialize retriever
    retriever = LlamaIndexMultimodalRetriever()
    
    # Create evaluation runner
    runner = EvalRunner(
        retriever=retriever,
        dataset_path="evaluation/eval_dataset.json",
        enable_langfuse=True,
    )
    
    # Run evaluation
    results = await runner.run_evaluation(
        experiment_name="duckyo_baseline_v1",
        max_queries=None,  # Run all queries
    )
    
    # Generate report
    report = runner.generate_report("duckyo_baseline_v1")
    
    # Save report
    report_path = Path("evaluation/results/duckyo_baseline_v1_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n Evaluation complete!")
    print(f"   Result file: evaluation/results/duckyo_baseline_v1.json")
    print(f"   Report file: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
"""
Evaluation result visualization
Generate charts to display evaluation results
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def generate_markdown_table(
    headers: List[str],
    rows: List[List[str]]
) -> str:
    """Generate Markdown table"""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join([" --- " for _ in headers]) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_metrics_comparison(
    results: Dict[str, Any],
    output_path: str = "evaluation/results/comparison.md"
) -> str:
    """
    Generate metrics comparison report
    """
    summary = results.get("summary", {})
    overall = summary.get("overall", {})
    
    lines = [
        "# Evaluation Results Comparison",
        "",
        "## Overall metrics",
        "",
    ]
    
    # Overall metrics table
    metrics_table = [
        ["Metric", "Mean", "Std", "Min", "Max"],
    ]
    
    for k in [1, 3, 5, 10]:
        if k in overall.get("recall_at_k", {}):
            r = overall["recall_at_k"][k]
            metrics_table.append([
                f"Recall@{k}",
                f"{r['mean']:.4f}",
                f"{r['std']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
            ])
    
    for k in [1, 3, 5, 10]:
        if k in overall.get("precision_at_k", {}):
            r = overall["precision_at_k"][k]
            metrics_table.append([
                f"Precision@{k}",
                f"{r['mean']:.4f}",
                f"{r['std']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
            ])
    
    for k in [1, 3, 5, 10]:
        if k in overall.get("ndcg_at_k", {}):
            r = overall["ndcg_at_k"][k]
            metrics_table.append([
                f"NDCG@{k}",
                f"{r['mean']:.4f}",
                f"{r['std']:.4f}",
                f"{r['min']:.4f}",
                f"{r['max']:.4f}",
            ])
    
    if "mrr" in overall:
        r = overall["mrr"]
        metrics_table.append([
            "MRR",
            f"{r['mean']:.4f}",
            f"{r['std']:.4f}",
            "-",
            "-",
        ])
    
    if "map" in overall:
        r = overall["map"]
        metrics_table.append([
            "MAP",
            f"{r['mean']:.4f}",
            f"{r['std']:.4f}",
            "-",
            "-",
        ])
    
    if "latency_ms" in overall:
        r = overall["latency_ms"]
        metrics_table.append([
            "Latency (ms)",
            f"{r['mean']:.2f}",
            f"{r['std']:.2f}",
            f"{r.get('p50', 0):.2f}",
            f"{r.get('p95', 0):.2f}",
        ])
    
    lines.append(generate_markdown_table(
        metrics_table[0],
        metrics_table[1:]
    ))
    
    # By query type
    lines.extend([
        "",
        "## By query type",
        "",
    ])
    
    for query_type, metrics in summary.get("by_query_type", {}).items():
        lines.extend([
            f"### {query_type.upper()}",
            "",
        ])
        
        type_table = [
            ["Metric", "Mean", "Std"],
        ]
        
        for k in [1, 3, 5, 10]:
            if k in metrics.get("recall_at_k", {}):
                r = metrics["recall_at_k"][k]
                type_table.append([f"Recall@{k}", f"{r['mean']:.4f}", f"{r['std']:.4f}"])
        
        if "mrr" in metrics:
            r = metrics["mrr"]
            type_table.append(["MRR", f"{r['mean']:.4f}", f"{r['std']:.4f}"])
        
        if "map" in metrics:
            r = metrics["map"]
            type_table.append(["MAP", f"{r['mean']:.4f}", f"{r['std']:.4f}"])
        
        lines.append(generate_markdown_table(type_table[0], type_table[1:]))
        lines.append("")
    
    # By difficulty
    lines.extend([
        "## By difficulty level",
        "",
    ])
    
    for difficulty, metrics in summary.get("by_difficulty", {}).items():
        lines.extend([
            f"### {difficulty.upper()}",
            "",
        ])
        
        diff_table = [
            ["Metric", "Mean", "Std"],
        ]
        
        for k in [1, 3, 5, 10]:
            if k in metrics.get("recall_at_k", {}):
                r = metrics["recall_at_k"][k]
                diff_table.append([f"Recall@{k}", f"{r['mean']:.4f}", f"{r['std']:.4f}"])
        
        if "mrr" in metrics:
            r = metrics["mrr"]
            diff_table.append(["MRR", f"{r['mean']:.4f}", f"{r['std']:.4f}"])
        
        lines.append(generate_markdown_table(diff_table[0], diff_table[1:]))
        lines.append("")
    
    content = "\n".join(lines)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f" Comparison report generated: {output_path}")
    return content


def generate_failure_analysis(
    results: Dict[str, Any],
    output_path: str = "evaluation/results/failure_analysis.md"
) -> str:
    """
    Generate failure case analysis
    """
    detailed = results.get("detailed_results", [])
    
    # Find failed queries
    failures = [r for r in detailed if not r.get("success", True)]
    low_recall = [r for r in detailed if r.get("metrics", {}).get("recall_at_k", {}).get(5, 1.0) < 0.5]
    
    lines = [
        "# Failure Case Analysis",
        "",
        f"**Total queries**: {len(detailed)}",
        f"**Failed queries**: {len(failures)}",
        f"**Low recall queries (Recall@5 < 0.5)**: {len(low_recall)}",
        "",
    ]
    
    if failures:
        lines.extend([
            "## Failed queries",
            "",
        ])
        for r in failures:
            lines.extend([
                f"### Query: {r.get('query', 'N/A')}",
                "",
                f"- **Type**: {r.get('query_type', 'N/A')}",
                f"- **Difficulty**: {r.get('difficulty', 'N/A')}",
                f"- **Error**: {r.get('error', 'Unknown')}",
                "",
            ])
    
    if low_recall:
        lines.extend([
            "## Low recall query analysis",
            "",
        ])
        for r in sorted(low_recall, key=lambda x: x.get("metrics", {}).get("recall_at_k", {}).get(5, 0)):
            lines.extend([
                f"### Query: {r.get('query', 'N/A')}",
                "",
                f"- **Type**: {r.get('query_type', 'N/A')}",
                f"- **Difficulty**: {r.get('difficulty', 'N/A')}",
                f"- **Expected results**: {r.get('expected_ids', [])}",
                f"- **Actual results**: {r.get('retrieved_ids', [])[:5]}",
                f"- **Recall@5**: {r.get('metrics', {}).get('recall_at_k', {}).get(5, 0):.4f}",
                "",
            ])
    
    content = "\n".join(lines)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f" Failure analysis generated: {output_path}")
    return content


def main():
    """Main function"""
    import sys
    
    # Read evaluation results
    result_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation/results/duckyo_baseline_v1.json"
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Generate report
    generate_metrics_comparison(results)
    generate_failure_analysis(results)
    
    print("\n All visualization reports generated!")


if __name__ == "__main__":
    main()

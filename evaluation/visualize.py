"""
评测结果可视化
生成图表展示评测结果
"""
import json
from pathlib import Path
from typing import Dict, List, Any


def generate_markdown_table(
    headers: List[str],
    rows: List[List[str]]
) -> str:
    """生成 Markdown 表格"""
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
    生成指标对比报告
    """
    summary = results.get("summary", {})
    overall = summary.get("overall", {})
    
    lines = [
        "# 评测结果对比",
        "",
        "## 总体指标",
        "",
    ]
    
    # 总体指标表格
    metrics_table = [
        ["指标", "均值", "标准差", "最小值", "最大值"],
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
    
    # 按查询类型
    lines.extend([
        "",
        "## 按查询类型",
        "",
    ])
    
    for query_type, metrics in summary.get("by_query_type", {}).items():
        lines.extend([
            f"### {query_type.upper()}",
            "",
        ])
        
        type_table = [
            ["指标", "均值", "标准差"],
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
    
    # 按难度
    lines.extend([
        "## 按难度级别",
        "",
    ])
    
    for difficulty, metrics in summary.get("by_difficulty", {}).items():
        lines.extend([
            f"### {difficulty.upper()}",
            "",
        ])
        
        diff_table = [
            ["指标", "均值", "标准差"],
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
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 对比报告已生成: {output_path}")
    return content


def generate_failure_analysis(
    results: Dict[str, Any],
    output_path: str = "evaluation/results/failure_analysis.md"
) -> str:
    """
    生成失败案例分析
    """
    detailed = results.get("detailed_results", [])
    
    # 找出失败的查询
    failures = [r for r in detailed if not r.get("success", True)]
    low_recall = [r for r in detailed if r.get("metrics", {}).get("recall_at_k", {}).get(5, 1.0) < 0.5]
    
    lines = [
        "# 失败案例分析",
        "",
        f"**总查询数**: {len(detailed)}",
        f"**失败查询**: {len(failures)}",
        f"**低召回查询 (Recall@5 < 0.5)**: {len(low_recall)}",
        "",
    ]
    
    if failures:
        lines.extend([
            "## 执行失败的查询",
            "",
        ])
        for r in failures:
            lines.extend([
                f"### Query: {r.get('query', 'N/A')}",
                "",
                f"- **类型**: {r.get('query_type', 'N/A')}",
                f"- **难度**: {r.get('difficulty', 'N/A')}",
                f"- **错误**: {r.get('error', 'Unknown')}",
                "",
            ])
    
    if low_recall:
        lines.extend([
            "## 低召回查询分析",
            "",
        ])
        for r in sorted(low_recall, key=lambda x: x.get("metrics", {}).get("recall_at_k", {}).get(5, 0)):
            lines.extend([
                f"### Query: {r.get('query', 'N/A')}",
                "",
                f"- **类型**: {r.get('query_type', 'N/A')}",
                f"- **难度**: {r.get('difficulty', 'N/A')}",
                f"- **期望结果**: {r.get('expected_ids', [])}",
                f"- **实际结果**: {r.get('retrieved_ids', [])[:5]}",
                f"- **Recall@5**: {r.get('metrics', {}).get('recall_at_k', {}).get(5, 0):.4f}",
                "",
            ])
    
    content = "\n".join(lines)
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 失败分析已生成: {output_path}")
    return content


def main():
    """主函数"""
    import sys
    
    # 读取评测结果
    result_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation/results/duckyo_baseline_v1.json"
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 生成报告
    generate_metrics_comparison(results)
    generate_failure_analysis(results)
    
    print("\n✅ 所有可视化报告已生成！")


if __name__ == "__main__":
    main()

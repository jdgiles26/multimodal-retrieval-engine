# Duckyo 多模态检索引擎评测体系

基于 Langfuse + LlamaIndex 的端到端评测框架

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Framework                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ EvalDataset │  │   Metrics   │  │ EvalRunner  │             │
│  │   Builder   │  │ Calculator  │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │              LlamaIndex Retriever                │           │
│  │  (Embedding + VectorStore + Rerank + Summary)    │           │
│  └──────────────────────────────────────────────────┘           │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────┐           │
│  │                  Langfuse                         │           │
│  │  (Traces, Scores, Datasets, Experiments)         │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 评测指标

### 准确性指标

| 指标 | 说明 | 公式 |
|------|------|------|
| **Recall@K** | 前K个结果中召回的相关文档比例 | `R@K = 相关文档在前K中的数量 / 相关文档总数` |
| **Precision@K** | 前K个结果的精确率 | `P@K = 相关文档在前K中的数量 / K` |
| **NDCG@K** | 归一化折损累积增益 | 考虑排序位置的相关性加权 |
| **MRR** | 平均倒数排名 | `MRR = 1 / 第一个相关文档的排名` |
| **MAP** | 平均精度均值 | 所有查询AP的平均值 |

### 效率指标

| 指标 | 说明 |
|------|------|
| **Latency (p50/p95/p99)** | 延迟分位数 |
| **Throughput (QPS)** | 每秒查询数 |

## 评测数据集

### 查询类型分布

- **Text Search**: 纯文本查询
- **Image Search**: 图搜图查询
- **Hybrid Search**: 图文混合查询

### 难度级别

- **Easy**: 城市名称、标签等精确匹配
- **Medium**: 地标、服装等概念匹配
- **Hard**: 心情、氛围等语义匹配

## 使用方法

### 1. 构建评测数据集

```bash
cd /root/.openclaw/workspace/duckyo/projects/multimodal-retrieval
python evaluation/eval_dataset.py
```

### 2. 运行完整评测

```bash
python evaluation/eval_runner.py
```

### 3. 运行基准测试

```bash
python evaluation/benchmark.py
```

### 4. 生成可视化报告

```bash
python evaluation/visualize.py evaluation/results/duckyo_baseline_v1.json
```

## 文件结构

```
evaluation/
├── __init__.py          # 模块初始化
├── eval_dataset.py      # 评测数据集构建
├── eval_metrics.py      # 评测指标计算
├── eval_runner.py       # 评测执行器
├── benchmark.py         # 基准测试
├── visualize.py         # 结果可视化
├── README.md            # 本文件
├── eval_dataset.json    # 生成的评测数据集
└── results/             # 评测结果
    ├── duckyo_baseline_v1.json
    ├── duckyo_baseline_v1_report.md
    ├── benchmark.json
    └── benchmark_report.md
```

## Langfuse 集成

评测结果会自动记录到 Langfuse：

- **Traces**: 每个查询的执行追踪
- **Scores**: 各项指标分数
- **Datasets**: 评测数据集版本管理
- **Experiments**: 不同实验对比

## 扩展指南

### 添加新的评测查询

编辑 `eval_dataset.py` 中的 `DuckyoEvalDatasetBuilder` 类：

```python
def build_custom_queries(self) -> List[QueryRelevance]:
    queries = []
    # 添加自定义查询...
    return queries
```

### 添加新的指标

编辑 `eval_metrics.py` 中的 `MetricsCalculator` 类：

```python
@staticmethod
def calculate_custom_metric(...) -> float:
    # 实现自定义指标
    pass
```

## 参考

- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/latest/optimizing/evaluation/evaluation/)
- [Langfuse Evaluation](https://langfuse.com/docs/evaluation)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

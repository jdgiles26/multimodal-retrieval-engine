# Duckyo Multimodal Retrieval Engine Evaluation System

End-to-end evaluation framework based on Langfuse + LlamaIndex

## Architecture

```
+-----------------------------------------------------------------+
|                        Evaluation Framework                      |
+-----------------------------------------------------------------+
|  +-------------+  +-------------+  +-------------+             |
|  | EvalDataset |  |   Metrics   |  | EvalRunner  |             |
|  |   Builder   |  | Calculator  |  |             |             |
|  +------+------+  +------+------+  +------+------+             |
|         |                |                |                      |
|         v                v                v                      |
|  +--------------------------------------------------+           |
|  |              LlamaIndex Retriever                |           |
|  |  (Embedding + VectorStore + Rerank + Summary)    |           |
|  +--------------------------------------------------+           |
|         |                                                        |
|         v                                                        |
|  +--------------------------------------------------+           |
|  |                  Langfuse                         |           |
|  |  (Traces, Scores, Datasets, Experiments)         |           |
|  +--------------------------------------------------+           |
+-----------------------------------------------------------------+
```

## Evaluation Metrics

### Accuracy Metrics

| Metric | Description | Formula |
|------|------|------|
| **Recall@K** | Proportion of relevant documents recalled in top K results | `R@K = relevant in top K / total relevant` |
| **Precision@K** | Precision of top K results | `P@K = relevant in top K / K` |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Relevance weighting considering ranking position |
| **MRR** | Mean Reciprocal Rank | `MRR = 1 / rank of first relevant` |
| **MAP** | Mean Average Precision | Average of AP across all queries |

### Efficiency Metrics

| Metric | Description |
|------|------|
| **Latency (p50/p95/p99)** | Latency percentiles |
| **Throughput (QPS)** | Queries per second |

## Evaluation Dataset

### Query Type Distribution

- **Text Search**: Pure text queries
- **Image Search**: Image-to-image queries
- **Hybrid Search**: Text-image hybrid queries

### Difficulty Levels

- **Easy**: Exact match for city names, tags, etc.
- **Medium**: Concept matching for landmarks, clothing, etc.
- **Hard**: Semantic matching for mood, atmosphere, etc.

## Usage

### 1. Build Evaluation Dataset

```bash
cd .
python evaluation/eval_dataset.py
```

### 2. Run Full Evaluation

```bash
python evaluation/eval_runner.py
```

### 3. Run Benchmark

```bash
python evaluation/benchmark.py
```

### 4. Generate Visualization Report

```bash
python evaluation/visualize.py evaluation/results/duckyo_baseline_v1.json
```

## File Structure

```
evaluation/
+-- __init__.py          # Module initialization
+-- eval_dataset.py      # Evaluation dataset construction
+-- eval_metrics.py      # Evaluation metrics calculation
+-- eval_runner.py       # Evaluation runner
+-- benchmark.py         # Benchmark
+-- visualize.py         # Result visualization
+-- README.md            # This file
+-- eval_dataset.json    # Generated evaluation dataset
+-- results/             # Evaluation results
    +-- duckyo_baseline_v1.json
    +-- duckyo_baseline_v1_report.md
    +-- benchmark.json
    +-- benchmark_report.md
```

## Langfuse Integration

Evaluation results are automatically recorded to Langfuse:

- **Traces**: Execution trace for each query
- **Scores**: Scores for each metric
- **Datasets**: Evaluation dataset version management
- **Experiments**: Comparison of different experiments

## Extension Guide

### Add New Evaluation Queries

Edit `eval_dataset.py` and find the `DuckyoEvalDatasetBuilder` class:

```python
def build_custom_queries(self) -> List[QueryRelevance]:
    queries = []
    # Add custom queries...
    return queries
```

### Add New Metrics

Edit `eval_metrics.py` and find the `MetricsCalculator` class:

```python
@staticmethod
def calculate_custom_metric(...) -> float:
    # Implement custom metrics
    pass
```

## References

- [LlamaIndex Evaluation](https://docs.llamaindex.ai/en/latest/optimizing/evaluation/evaluation/)
- [Langfuse Evaluation](https://langfuse.com/docs/evaluation)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

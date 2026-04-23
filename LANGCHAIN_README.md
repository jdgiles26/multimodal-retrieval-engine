# Multimodal Retrieval System - LangChain Version

A multimodal retrieval engine based on **LangChain + Milvus + Qwen3-VL**.

## Architecture

```
+-------------------------------------------------------------+
|              LangChain Multimodal RAG                       |
+-------------------------------------------------------------+
|  Input Layer                                                |
|  |-- Document Loaders (image / text)                        |
|  \-- MultiVectorRetriever                                   |
+-------------------------------------------------------------+
|  Processing Layer                                           |
|  |-- Qwen3VLEmbeddings (2560-dim)                           |
|  |   \-- Unified text + image encoding                      |
|  \-- Milvus VectorStore                                     |
|      \-- HNSW index + cosine similarity                     |
+-------------------------------------------------------------+
|  Retrieval Layer                                            |
|  |-- VectorStoreRetriever                                   |
|  \-- Reranker (gte-rerank)                                  |
+-------------------------------------------------------------+
|  API Layer (FastAPI)                                        |
|  \-- RESTful endpoints                                      |
+-------------------------------------------------------------+
```

## Core Components

### 1. Qwen3VLEmbeddings

```python
from core.langchain_engine import Qwen3VLEmbeddings

embeddings = Qwen3VLEmbeddings(
    api_key="sk-xxx",
    model="qwen3-vl-embedding",
    dimension=2560
)

# Text embedding
vec = embeddings.embed_query("Tokyo tower night view")

# Image embedding
vec = embeddings.embed_image("https://.../photo.jpg")

# Text-image fusion
vec = embeddings.embed_fusion(
    text="Tokyo tower night view",
    image_url="https://.../photo.jpg"
)
```

### 2. MultimodalRetriever

```python
from core.langchain_engine import MultimodalRetriever

retriever = MultimodalRetriever(
    api_key="sk-xxx",
    milvus_uri="http://localhost:19530",
    collection_name="duckyo_travel_memories"
)

# Add data
memories = [
    {
        "content": "Tokyo tower night view, wearing Hanfu",
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["Hanfu", "night view"]
    }
]
retriever.add_travel_memories(memories)

# Text-to-image search
results = retriever.search_by_text("Tokyo tower night view Hanfu", top_k=5)

# Image-to-image search
results = retriever.search_by_image("https://.../query.jpg", top_k=5)

# Text-image hybrid search
results = retriever.search_by_fusion(
    text="Hanfu",
    image_url="https://.../ref.jpg",
    top_k=5
)
```

## Quick Start

### Install Dependencies

```bash
pip install langchain langchain-milvus
pip install pymilvus dashscope
```

### Start Milvus

```bash
docker-compose up -d
```

### Run Tests

```bash
python -c "
from core.langchain_engine import MultimodalRetriever
import os

retriever = MultimodalRetriever(api_key=os.getenv('DASHSCOPE_API_KEY'))

# Add test data
retriever.add_travel_memories([
    {'content': 'Tokyo tower night view', 'city': 'Tokyo', 'date': '2026-03-06'},
    {'content': 'Sydney Opera House', 'city': 'Sydney', 'date': '2026-03-09'},
])

# Search
results = retriever.search_by_text('Tokyo')
for r in results:
    print(f'{r.metadata[\"city\"]}: {r.page_content}')
"
```

## Comparison: Custom vs LangChain

| Feature | Custom Version | LangChain Version |
|---------|---------------|-------------------|
| Code volume | More | Less |
| Flexibility | High | Medium |
| Ecosystem integration | Custom development needed | Rich |
| Maintenance cost | High | Low |
| Production ready | Extra work needed | Closer to production |
| Learning curve | Low | Medium |

## Two Approaches Coexist

The project provides two implementations:

1. **Custom version** (`core/engine.py`)
   - Full control over all details
   - Suitable for deep customization

2. **LangChain version** (`core/langchain_engine.py`)
   - Standardized interfaces
   - Easy integration with other tools

Choose or mix based on your needs!

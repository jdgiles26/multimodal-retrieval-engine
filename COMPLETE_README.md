# Multimodal Retrieval System - Complete Solution

A multimodal retrieval engine based on **Qwen3-VL Embedding + Milvus**, supporting three implementation approaches.

## Project Overview

**Core Features**:
- Text-to-Image search
- Image-to-Image search
- Multi-Image-to-Image search
- Text+Image hybrid search
- Two-stage retrieval (vector coarse filter + rerank fine-tuning)

**Tech Stack**:
- **Embedding**: qwen3-vl-embedding (2560-dim)
- **Rerank**: gte-rerank
- **Vector DB**: Milvus
- **API**: FastAPI

---

## Three Implementation Approaches

### Approach 1: Custom Version (Recommended)

**Location**: `core/engine.py`

**Characteristics**:
- Full control over all details
- Clean and understandable code
- No framework dependencies
- Optimal performance

**Usage**:
```python
from core import get_engine

engine = get_engine()

# Text-to-image search
results = await engine.text_search("Tokyo tower night view", top_k=10)

# Image-to-image search
results = await engine.image_search("https://.../photo.jpg", top_k=10)

# Text+image hybrid search
results = await engine.hybrid_search(
    text="Hanfu",
    image="https://.../ref.jpg",
    top_k=10
)
```

---

### Approach 2: LangChain Version

**Location**: `core/langchain_engine.py`

**Characteristics**:
- Richest ecosystem
- Standardized interfaces
- Easy integration with other tools
- Requires custom multimodal support

**Usage**:
```python
from core.langchain_engine import MultimodalRetriever

retriever = MultimodalRetriever(api_key="sk-xxx")

# Add data
retriever.add_travel_memories([...])

# Search
results = retriever.search_by_text("Tokyo tower night view", top_k=10)
```

---

### Approach 3: LlamaIndex Version

**Location**: `core/llamaindex_engine.py`

**Characteristics**:
- Native multimodal support
- Most concise code
- Automatic text+image handling
- Focused on retrieval scenarios

**Usage**:
```python
from core.llamaindex_engine import LlamaIndexMultimodalRetriever

retriever = LlamaIndexMultimodalRetriever(api_key="sk-xxx")

# Add data (auto-detects images)
retriever.add_travel_memories([...])

# Search
results = retriever.search("Tokyo tower night view", top_k=10)
```

---

## Approach Comparison

| Feature | Custom | LangChain | LlamaIndex |
|---------|--------|-----------|------------|
| Code volume | Medium | Less | **Least** |
| Multimodal support | Good | Good | **Excellent** |
| Ecosystem richness | Limited | **Rich** | Good |
| Learning curve | Low | Medium | Medium |
| Production ready | Yes | Yes | Yes |
| Flexibility | **High** | Medium | Medium |
| Performance | **Best** | Good | Good |

**Recommendations**:
- **Rapid prototyping**: Custom version
- **Production deployment**: LangChain or LlamaIndex
- **Multimodal RAG**: LlamaIndex (native support)
- **Complex AI applications**: LangChain (richest ecosystem)

---

## Quick Start

### 1. Install Dependencies

```bash
# Base dependencies
pip install -r requirements.txt

# LangChain version
pip install langchain langchain-milvus

# LlamaIndex version
pip install llama-index llama-index-vector-stores-milvus
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in DASHSCOPE_API_KEY
```

### 3. Start Milvus

```bash
docker-compose up -d
```

### 4. Run Tests

```bash
# Custom version test
python scripts/test_api.py

# Framework comparison test
python scripts/compare_frameworks.py

# Start API
python -m api.main
```

---

## API Endpoints

Access at: `http://localhost:8000/docs`

### Core Endpoints

```bash
# Unified search
POST /search
{
  "query_text": "Tokyo tower night view Hanfu",
  "filters": {"city": "Tokyo"},
  "top_k": 10
}

# Text-to-image search
POST /search/text?q=Tokyo&city=Tokyo&top_k=10

# Image-to-image search
POST /search/image?image_url=https://...&top_k=10

# Batch ingestion
POST /ingest
{
  "items": [...]
}
```

---

## Project Structure

```
multimodal-retrieval/
|-- api/
|   \-- main.py                    # FastAPI application
|-- core/
|   |-- __init__.py
|   |-- embedder.py                # Qwen3-VL Embedding
|   |-- reranker.py                # gte-rerank
|   |-- milvus_client.py           # Milvus client
|   |-- engine.py                  # Custom retrieval engine
|   |-- langchain_engine.py        # LangChain version
|   \-- llamaindex_engine.py       # LlamaIndex version
|-- scripts/
|   |-- test_api.py                # API testing
|   |-- ingest_travel_data.py      # Data import
|   \-- compare_frameworks.py      # Framework comparison
|-- tests/
|   \-- test_core.py               # Unit tests
|-- config/
|   \-- settings.py                # Configuration management
|-- docker-compose.yml             # Milvus deployment
|-- requirements.txt               # Dependencies
|-- README.md                      # Documentation
|-- LANGCHAIN_README.md            # LangChain docs
\-- COMPLETE_README.md             # This file
```

---

## Advanced Features

### Hybrid Retrieval (BM25 + Vector)

```python
from langchain_milvus import BM25BuiltInFunction

vectorstore = Milvus(
    embedding_function=embeddings,
    builtin_functions=[BM25BuiltInFunction()],
    # ...
)
```

### Reranker Integration

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

compressor = CrossEncoderReranker(model="gte-rerank")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Multimodal Fusion Encoding

```python
# Fuse text and images into a single vector
embedding = await embedder.embed_fusion(
    text="Tokyo tower night view",
    images=["https://.../photo1.jpg", "https://.../photo2.jpg"]
)
```

---

## Cost Estimation

| Service | Unit Price | Estimated Usage | Monthly Cost |
|---------|-----------|-----------------|--------------|
| qwen3-vl-embedding | $0.258 / 1K images | 10K images | $2.58 |
| gte-rerank | $0.015 / 1K tokens | 100K tokens | $15.00 |
| Milvus | Free | - | $0 |
| **Total** | | | **~$18/month** |

---

## Next Steps

1. **Start full services**
   ```bash
   ./deploy.sh
   ```

2. **Import travel data**
   ```bash
   python cli.py ingest --dir /path/to/photos
   ```

3. **Test retrieval**
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query_text": "Tokyo tower night view Hanfu"}'
   ```

---

## References

- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Documentation](https://python.langchain.com)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)

---

_Built with care by Duckyo_

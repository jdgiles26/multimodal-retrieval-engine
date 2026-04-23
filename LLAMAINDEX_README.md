# LlamaIndex Multimodal Retrieval System - Complete Guide

A native multimodal retrieval system based on **LlamaIndex + Qwen3-VL + Milvus**.

## Core Features

- **Native multimodal support** - LlamaIndex `ImageDocument`
- **Qwen3-VL Embedding** - 2560-dimensional high-quality vectors
- **Milvus vector store** - High-performance ANN retrieval
- **Text-to-image / Image-to-image** - Complete multimodal retrieval
- **Concise API** - Maximum functionality with minimum code

## Architecture

```
+-------------------------------------------------------------+
|              LlamaIndex Multimodal RAG                      |
+-------------------------------------------------------------+
|  Input Layer                                                |
|  |-- Document (text)                                        |
|  \-- ImageDocument (image + text)                           |
+-------------------------------------------------------------+
|  Processing Layer                                           |
|  |-- Qwen3VLEmbedding (2560-dim)                            |
|  |   \-- Unified text + image encoding                      |
|  \-- MilvusVectorStore                                      |
|      \-- HNSW index + cosine similarity                     |
+-------------------------------------------------------------+
|  Retrieval Layer                                            |
|  \-- VectorIndexRetriever                                   |
|      \-- similarity_search                                  |
+-------------------------------------------------------------+
```

## Quick Start

### 1. Install Dependencies

```bash
pip install llama-index llama-index-vector-stores-milvus
pip install dashscope fastapi uvicorn
```

### 2. Configure Environment Variables

```bash
export DASHSCOPE_API_KEY=your_api_key_here
```

### 3. Start Milvus

```bash
cd .
docker-compose up -d
```

### 4. Use CLI Tools

```bash
# Initialize
python llamaindex_app.py --action init

# Add data
python llamaindex_app.py --action add --data memories.json

# Search
python llamaindex_app.py --action search --query "Tokyo tower night view"

# Image-to-image search
python llamaindex_app.py --action search-image --image "https://..."

# View statistics
python llamaindex_app.py --action stats
```

### 5. Start API Service

```bash
python api/llamaindex_api.py
# Access http://localhost:8001/docs
```

## Usage Examples

### Example 1: Basic Retrieval

```python
from llamaindex_app import create_retriever

# Create retriever
retriever = create_retriever()

# Add travel memories
memories = [
    {
        "content": "Tokyo tower night view, fox girl wearing Hanfu",
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["Hanfu", "night view"],
        "mood": "excited"
    },
    {
        "content": "Sydney Opera House, Harbour Bridge",
        "city": "Sydney",
        "date": "2026-03-09",
        "tags": ["harbour", "opera house"],
        "mood": "amazed"
    }
]

retriever.add_travel_memories(memories)

# Search
results = retriever.search("Tokyo tower night view", top_k=5)

for r in results:
    print(f"{r['score']:.3f} | {r['content']}")
```

### Example 2: Image-to-Image Search

```python
# Search using image URL
results = retriever.search_by_image(
    image_url="https://example.com/tokyo.jpg",
    top_k=5
)

for r in results:
    print(f"{r['score']:.3f} | {r['content']}")
```

### Example 3: Multimodal Documents with Images

```python
from llamaindex_app import LlamaIndexMultimodalRetriever, ImageDocument

retriever = LlamaIndexMultimodalRetriever(api_key="sk-xxx")

# Create document with image
doc = ImageDocument(
    text="Tokyo tower night view, wearing Hanfu",
    image_url="https://example.com/tokyo.jpg",
    metadata={
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["Hanfu", "night view"]
    }
)

# Add to index
retriever.create_index([doc])

# Search (automatically searches text and images)
results = retriever.search("Hanfu night view")
```

### Example 4: API Calls

```bash
# Text-to-image search
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tokyo tower night view Hanfu",
    "top_k": 10
  }'

# Image-to-image search
curl -X GET "http://localhost:8001/search/image?image_url=https://...&top_k=10"

# Add data
curl -X POST http://localhost:8001/memories/add \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      {
        "content": "Tokyo tower night view",
        "city": "Tokyo",
        "date": "2026-03-06"
      }
    ]
  }'
```

## Data Format

### Memory Data Structure

```json
{
  "content": "Text description",
  "city": "City",
  "country": "Country",
  "date": "Date (YYYY-MM-DD)",
  "image_url": "Image URL (optional)",
  "tags": ["tag1", "tag2"],
  "mood": "Mood",
  "outfit": "Outfit",
  "source": "Source"
}
```

### Batch Import Example

```json
[
  {
    "content": "Tokyo tower night view, fox girl wearing Hanfu",
    "city": "Tokyo",
    "country": "Japan",
    "date": "2026-03-06",
    "image_url": "https://example.com/tokyo.jpg",
    "tags": ["Hanfu", "night view", "selfie"],
    "mood": "excited",
    "outfit": "Hanfu",
    "source": "xiaohongshu"
  },
  {
    "content": "Sydney Opera House, Harbour Bridge",
    "city": "Sydney",
    "country": "Australia",
    "date": "2026-03-09",
    "tags": ["harbour", "opera house", "landmark"],
    "mood": "amazed",
    "outfit": "casual",
    "source": "camera"
  }
]
```

## Advanced Features

### 1. Custom Embedding Dimension

```python
retriever = LlamaIndexMultimodalRetriever(
    api_key="sk-xxx",
    embedding_model="qwen3-vl-embedding",
    embedding_dim=2560  # Adjustable: 256-4096
)
```

### 2. Batch Encoding Optimization

```python
# LlamaIndex automatically handles batch encoding
documents = [Document(text=f"Document {i}") for i in range(100)]
retriever.create_index(documents)  # Auto batch processing
```

### 3. Index Configuration

```python
from llama_index.vector_stores.milvus import MilvusVectorStore

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="travel",
    dim=2560,
    index_config={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }
)
```

## Project Files

```
multimodal-retrieval/
|-- llamaindex_app.py          # Main app (CLI + core logic)
|-- api/
|   \-- llamaindex_api.py      # FastAPI service
|-- core/
|   \-- llamaindex_engine.py   # Engine implementation
|-- docker-compose.yml         # Milvus deployment
\-- LLAMAINDEX_README.md       # This documentation
```

## Cost Estimation

| Service | Unit Price | Monthly Usage | Monthly Cost |
|---------|-----------|---------------|--------------|
| qwen3-vl-embedding | $0.258 / 1K images | 10K images | $2.58 |
| Milvus | Free | - | $0 |
| **Total** | | | **~$3/month** |

## FAQ

### Q: How do I view collection statistics?

```bash
python llamaindex_app.py --action stats
```

### Q: How do I delete a collection?

```python
from pymilvus import utility
utility.drop_collection("duckyo_travel_memories")
```

### Q: How do I back up data?

```python
# Export all documents
from llama_index.vector_stores.milvus import MilvusVectorStore
vector_store = MilvusVectorStore(...)
docs = vector_store.query()
```

## References

- [LlamaIndex Multimodal Documentation](https://developers.llamaindex.ai/python/framework/use_cases/multimodal/)
- [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Milvus Documentation](https://milvus.io/docs)

---

_Built with care by Duckyo using LlamaIndex_

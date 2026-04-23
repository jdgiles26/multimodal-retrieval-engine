# Multimodal Retrieval Engine

A hybrid vector search system built on Qwen3-VL Embedding and Qwen3-VL Reranker, supporting text, image, and video search across a unified embedding space.

## Key Features

- **Multimodal Embedding**: Qwen3-VL natively supports text, images, and videos in a single model
- **Fusion Mode**: Combines text and images into a single joint vector representation
- **Two-Stage Retrieval**: Vector coarse filtering followed by multimodal reranking for precision
- **Milvus Vector Database**: High-performance Approximate Nearest Neighbors (ANN) indexing
- **FastAPI**: RESTful API with automatic OpenAPI documentation
- **CLI Tools**: Command-line interface for common operations

## Quick Start

### Option 1: One-Click Deployment (Recommended)
```bash
cd .
./deploy.sh
```

### Option 2: Manual Deployment
```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env and fill in DASHSCOPE_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Milvus
docker-compose up -d

# 4. Initialize collection
python cli.py init

# 5. Start API
python -m api.main
```

### Option 3: Docker Deployment
```bash
docker-compose up -d
```

## API Endpoints

Access interactive docs at: `http://localhost:8000/docs`

### Unified Search
`POST /search`
```json
{
  "query_text": "corn, indy",
  "filters": {"city": "Indianapolis"},
  "top_k": 10,
  "use_rerank": true
}
```

### Text Search
`POST /search/text?q=Tokyo tower night view&city=Tokyo&top_k=10`

### Image Search
`POST /search/image?image_url=https://example.com/photo.jpg&top_k=10`

### Multi-Image Search
`POST /search/images?image_urls=["url1","url2"]&top_k=10`

### Batch Ingestion
`POST /ingest`
```json
{
  "items": [
    {
      "id": "photo_001",
      "type": "image",
      "content": "this is content photo",
      "file_url": "https://example.com/USA.jpg",
      "city": "Indianapolis",
      "country": "USA",
      "date": "2026-04-08",
      "tags": ["corn", "basketball"],
      "mood": "hometown"
    }
  ]
}
```

## CLI Usage

```bash
# Initialize database
python cli.py init

# Text search
python cli.py search "USA" --city Indianapolis

# Image search
python cli.py search-image https://example.com/photo.jpg

# Import photos
python cli.py ingest --dir /path/to/photos

# View statistics
python cli.py stats
```

## Docker Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Service Ports:**
- API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Milvus: `http://localhost:19530`
- Attu (Admin): `http://localhost:3000`

## Architecture

View the system architecture diagram: [diagrams/tech-stack.excalidraw](diagrams/tech-stack.excalidraw)

The system consists of five layers:
1. **Clients** - Web UI, CLI, cURL/SDK
2. **API Gateway** - FastAPI with auto-generated OpenAPI docs
3. **Core Services** - Qwen3-VL Embedding, Search Engine, Rerank Service
4. **Data Layer** - Milvus vector database, MinIO object storage
5. **Infrastructure** - Docker Compose deployment

## Search Pipeline

View the search flow diagram: [diagrams/search-flow.excalidraw](diagrams/search-flow.excalidraw)

1. Receive query (text, image, or video)
2. Encode query with Qwen3-VL Embedding (2560-dim)
3. Perform ANN search in Milvus (retrieve Top-K x 3)
4. Optional rerank with gte-rerank / qwen3-rerank
5. Format and return top-K results

## UI Wireframes

View the interface wireframes: [diagrams/wireframes.excalidraw](diagrams/wireframes.excalidraw)

Two primary interfaces are provided:
- **FastAPI Docs** - Interactive Swagger UI for API exploration and testing
- **Search Portal** - Search bar with filters and ranked result cards

## Project Structure

```
multimodal-retrieval/
|-- api/
|   \-- main.py              # FastAPI application
|-- cli.py                   # CLI tools
|-- config/
|   \-- settings.py          # Configuration management
|-- core/
|   |-- embedder.py          # Qwen3-VL Embedding
|   |-- reranker.py          # Qwen3-VL Reranker
|   |-- milvus_client.py     # Milvus client
|   \-- engine.py            # Search engine
|-- scripts/
|   |-- test_api.py          # API testing
|   \-- ingest_travel_data.py # Bulk data import
|-- tests/
|   \-- test_core.py         # Unit tests
|-- deploy.sh                # One-click deployment
|-- start.sh                 # Startup script
|-- docker-compose.yml       # Docker Compose
|-- Dockerfile               # API Docker image
|-- requirements.txt         # Python dependencies
|-- .env.example             # Environment template
\-- README.md                # This file
```

## Cost Estimation

| Service | Price per Unit | Monthly Usage | Monthly Cost |
|---------|---------------|---------------|--------------|
| Qwen3-VL Embedding | $0.258 / 1K images | 10,000 images | $2.58 |
| Qwen3-VL Reranker | $0.015 / 1K tokens | 100,000 tokens | $15.00 |
| Milvus | Free | - | $0 |
| **Total** | | | **~$18/month** |

## Example Code

### Python SDK
```python
from core import get_engine

engine = get_engine()

# Text search
results = await engine.text_search(
    query="",
    filters={"city": "USA"},
    top_k=10
)

# Image search
results = await engine.image_search(
    image_url="https://example.com/photo.jpg",
    top_k=10
)

# Multi-image search
results = await engine.multi_image_search(
    image_urls=["url1", "url2", "url3"],
    top_k=10
)

# Hybrid (text + image) search
results = await engine.hybrid_search(
    text="textext",
    image="https://example.com/ref.jpg",
    top_k=10
)
```

### cURL
```bash
# Text search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{ "query_text": "txtxtxtxt", "top_k": 10 }'

# Image search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{ "query_image": "https://example.com/photo.jpg", "top_k": 10 }'
```

## Common Commands

```bash
# One-click deployment
./deploy.sh

# Start services
./start.sh

# CLI help
python cli.py --help

# Run tests
python scripts/test_api.py
pytest tests/

# View logs
docker-compose logs -f

# Enter Milvus container
docker-compose exec milvus-standalone bash
```

## FAQ

**Q: How do I handle local image files?**
A: Upload local images to a publicly accessible URL (MinIO, OSS, Imgur, etc.) before indexing.

**Q: Is the reranker required?**
A: No. Set `use_rerank: false` to skip reranking and reduce API costs.

**Q: How do I batch import existing photos?**
A: Use `python cli.py ingest --dir /path/to/photos` or call the `/ingest` API.

**Q: How do I view collection statistics?**
A: Access `http://localhost:8000/collection/stats` or run `python cli.py stats`.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific tests
pytest tests/test_core.py -v

# API testing
python scripts/test_api.py
```

## License

MIT License

jdgiles - dabbledablle

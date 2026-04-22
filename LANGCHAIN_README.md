# 🦊 Duckyo 多模态检索系统 - LangChain 版本

基于 **LangChain + Milvus + Qwen3-VL** 的多模态检索引擎

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LangChain 多模态 RAG                      │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                      │
│  ├── Document Loaders (图片/文本)                           │
│  └── MultiVectorRetriever                                    │
├─────────────────────────────────────────────────────────────┤
│  处理层                                                      │
│  ├── Qwen3VLEmbeddings (2560维)                            │
│  │   └── 统一编码文本+图片                                  │
│  └── Milvus VectorStore                                      │
│      └── HNSW索引 + 余弦相似度                              │
├─────────────────────────────────────────────────────────────┤
│  检索层                                                      │
│  ├── VectorStoreRetriever                                    │
│  └── Reranker (gte-rerank)                                  │
├─────────────────────────────────────────────────────────────┤
│  API 层 (FastAPI)                                           │
│  └── RESTful 接口                                           │
└─────────────────────────────────────────────────────────────┘
```

## 📦 核心组件

### 1. Qwen3VLEmbeddings

```python
from core.langchain_engine import Qwen3VLEmbeddings

embeddings = Qwen3VLEmbeddings(
    api_key="sk-xxx",
    model="qwen3-vl-embedding",
    dimension=2560
)

# 文本编码
vec = embeddings.embed_query("东京塔夜景")

# 图片编码
vec = embeddings.embed_image("https://.../photo.jpg")

# 图文融合
vec = embeddings.embed_fusion(
    text="东京塔夜景",
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

# 添加数据
memories = [
    {
        "content": "东京塔夜景, 穿着汉服",
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["汉服", "夜景"]
    }
]
retriever.add_travel_memories(memories)

# 文搜图
results = retriever.search_by_text("东京塔夜景汉服", top_k=5)

# 图搜图
results = retriever.search_by_image("https://.../query.jpg", top_k=5)

# 图文混合搜索
results = retriever.search_by_fusion(
    text="汉服",
    image_url="https://.../ref.jpg",
    top_k=5
)
```

## 🚀 快速开始

### 安装依赖

```bash
pip install langchain langchain-milvus
pip install pymilvus dashscope
```

### 启动 Milvus

```bash
docker-compose up -d
```

### 运行测试

```bash
python -c "
from core.langchain_engine import MultimodalRetriever
import os

retriever = MultimodalRetriever(api_key=os.getenv('DASHSCOPE_API_KEY'))

# 添加测试数据
retriever.add_travel_memories([
    {'content': '东京塔夜景', 'city': 'Tokyo', 'date': '2026-03-06'},
    {'content': '悉尼歌剧院', 'city': 'Sydney', 'date': '2026-03-09'},
])

# 搜索
results = retriever.search_by_text('东京塔')
for r in results:
    print(f'{r.metadata[\"city\"]}: {r.page_content}')
"
```

## 📊 对比：自定义 vs LangChain

| 特性 | 自定义版本 | LangChain 版本 |
|------|-----------|---------------|
| 代码量 | 较多 | 较少 |
| 灵活性 | 高 | 中 |
| 生态集成 | 需自行开发 | 丰富 |
| 维护成本 | 高 | 低 |
| 生产就绪 | 需额外工作 | 更接近生产 |
| 学习曲线 | 低 | 中 |

## 🔧 两种方案并存

当前项目提供两种实现：

1. **自定义版本** (`core/engine.py`)
   - 完全控制所有细节
   - 适合深度定制

2. **LangChain 版本** (`core/langchain_engine.py`)
   - 标准化接口
   - 易于集成其他工具

你可以根据需求选择或混合使用！
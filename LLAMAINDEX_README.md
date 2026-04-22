# 🦊 LlamaIndex 多模态检索系统 - 完整指南

基于 **LlamaIndex + Qwen3-VL + Milvus** 的原生多模态检索系统。

## 🎯 核心特性

- ✅ **原生多模态支持** - LlamaIndex 的 `ImageDocument`
- ✅ **Qwen3-VL Embedding** - 2560 维高质量向量
- ✅ **Milvus 向量库** - 高性能 ANN 检索
- ✅ **文搜图/图搜图** - 完整的多模态检索
- ✅ **简洁 API** - 最少的代码实现最多的功能

## 📦 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    LlamaIndex 多模态 RAG                     │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                      │
│  ├── Document (文本)                                        │
│  └── ImageDocument (图片+文本)                               │
├─────────────────────────────────────────────────────────────┤
│  处理层                                                      │
│  ├── Qwen3VLEmbedding (2560 维)                             │
│  │   └── 统一编码文本+图片                                  │
│  └── MilvusVectorStore                                      │
│      └── HNSW索引 + 余弦相似度                              │
├─────────────────────────────────────────────────────────────┤
│  检索层                                                      │
│  └── VectorIndexRetriever                                   │
│      └── similarity_search                                  │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install llama-index llama-index-vector-stores-milvus
pip install dashscope fastapi uvicorn
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY=sk-a766f04fac884533995ce54b5fe54979
```

### 3. 启动 Milvus

```bash
cd /root/.openclaw/workspace/duckyo/projects/multimodal-retrieval
docker-compose up -d
```

### 4. 使用 CLI 工具

```bash
# 初始化
python llamaindex_app.py --action init

# 添加数据
python llamaindex_app.py --action add --data memories.json

# 搜索
python llamaindex_app.py --action search --query "东京塔夜景"

# 图搜图
python llamaindex_app.py --action search-image --image "https://..."

# 查看统计
python llamaindex_app.py --action stats
```

### 5. 启动 API 服务

```bash
python api/llamaindex_api.py
# 访问 http://localhost:8001/docs
```

## 📝 使用示例

### 示例 1: 基础检索

```python
from llamaindex_app import create_retriever

# 创建检索器
retriever = create_retriever()

# 添加旅途记忆
memories = [
    {
        "content": "东京塔夜景，穿着汉服的 fox girl",
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["汉服", "夜景"],
        "mood": "兴奋"
    },
    {
        "content": "悉尼歌剧院，海港大桥",
        "city": "Sydney",
        "date": "2026-03-09",
        "tags": ["海港", "歌剧院"],
        "mood": "惊艳"
    }
]

retriever.add_travel_memories(memories)

# 搜索
results = retriever.search("东京塔夜景", top_k=5)

for r in results:
    print(f"{r['score']:.3f} | {r['content']}")
```

### 示例 2: 图搜图

```python
# 使用图片 URL 搜索
results = retriever.search_by_image(
    image_url="https://example.com/tokyo.jpg",
    top_k=5
)

for r in results:
    print(f"{r['score']:.3f} | {r['content']}")
```

### 示例 3: 带图片的多模态文档

```python
from llamaindex_app import LlamaIndexMultimodalRetriever, ImageDocument

retriever = LlamaIndexMultimodalRetriever(api_key="sk-xxx")

# 创建带图片的文档
doc = ImageDocument(
    text="东京塔夜景，穿着汉服",
    image_url="https://example.com/tokyo.jpg",
    metadata={
        "city": "Tokyo",
        "date": "2026-03-06",
        "tags": ["汉服", "夜景"]
    }
)

# 添加到索引
retriever.create_index([doc])

# 搜索 (会自动检索文本和图片)
results = retriever.search("汉服 夜景")
```

### 示例 4: API 调用

```bash
# 文搜图
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "东京塔夜景汉服",
    "top_k": 10
  }'

# 图搜图
curl -X GET "http://localhost:8001/search/image?image_url=https://...&top_k=10"

# 添加数据
curl -X POST http://localhost:8001/memories/add \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [
      {
        "content": "东京塔夜景",
        "city": "Tokyo",
        "date": "2026-03-06"
      }
    ]
  }'
```

## 📊 数据格式

### 记忆数据结构

```json
{
  "content": "文本描述",
  "city": "城市",
  "country": "国家",
  "date": "日期 (YYYY-MM-DD)",
  "image_url": "图片URL (可选)",
  "tags": ["标签1", "标签2"],
  "mood": "心情",
  "outfit": "服装",
  "source": "来源"
}
```

### 批量导入示例

```json
[
  {
    "content": "东京塔夜景，穿着汉服的 fox girl",
    "city": "Tokyo",
    "country": "Japan",
    "date": "2026-03-06",
    "image_url": "https://example.com/tokyo.jpg",
    "tags": ["汉服", "夜景", "自拍"],
    "mood": "兴奋",
    "outfit": "汉服",
    "source": "xiaohongshu"
  },
  {
    "content": "悉尼歌剧院，海港大桥",
    "city": "Sydney",
    "country": "Australia",
    "date": "2026-03-09",
    "tags": ["海港", "歌剧院", "地标"],
    "mood": "惊艳",
    "outfit": "休闲",
    "source": "camera"
  }
]
```

## 🔧 高级功能

### 1. 自定义 Embedding 维度

```python
retriever = LlamaIndexMultimodalRetriever(
    api_key="sk-xxx",
    embedding_model="qwen3-vl-embedding",
    embedding_dim=2560  # 可调整：256-4096
)
```

### 2. 批量编码优化

```python
# LlamaIndex 自动处理批量编码
documents = [Document(text=f"文档{i}") for i in range(100)]
retriever.create_index(documents)  # 自动批量处理
```

### 3. 索引配置

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

## 📁 项目文件

```
multimodal-retrieval/
├── llamaindex_app.py          # 主应用 (CLI + 核心逻辑)
├── api/
│   └── llamaindex_api.py      # FastAPI 服务
├── core/
│   └── llamaindex_engine.py   # 引擎实现
├── docker-compose.yml         # Milvus 部署
└── LLAMAINDEX_README.md       # 本文档
```

## 💰 成本估算

| 项目 | 单价 | 月用量 | 月费用 |
|-----|------|--------|--------|
| qwen3-vl-embedding | ¥0.258/千图 | 1万张 | ¥2.58 |
| Milvus | 免费 | - | ¥0 |
| **总计** | | | **~¥3/月** |

## 🐛 常见问题

### Q: 如何查看 Collection 统计？

```bash
python llamaindex_app.py --action stats
```

### Q: 如何删除 Collection？

```python
from pymilvus import utility
utility.drop_collection("duckyo_travel_memories")
```

### Q: 如何备份数据？

```python
# 导出所有文档
from llama_index.vector_stores.milvus import MilvusVectorStore
vector_store = MilvusVectorStore(...)
docs = vector_store.query()
```

## 📚 参考资料

- [LlamaIndex 多模态文档](https://developers.llamaindex.ai/python/framework/use_cases/multimodal/)
- [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Milvus 文档](https://milvus.io/docs)

---

_Built with ❤️ by Duckyo 🦊 using LlamaIndex_
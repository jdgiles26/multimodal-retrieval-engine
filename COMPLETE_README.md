# 🦊 Duckyo 多模态检索系统 - 完整方案

基于 **Qwen3-VL Embedding + Milvus** 的多模态检索引擎，支持三种实现方案。

## 🎯 项目概述

**核心功能**:
- ✅ 文搜图 (Text-to-Image)
- ✅ 图搜图 (Image-to-Image)  
- ✅ 多图搜图 (Multi-Image-to-Image)
- ✅ 图文混合搜索 (Text+Image)
- ✅ 两阶段检索 (向量粗筛 + Rerank精排)

**技术栈**:
- **Embedding**: qwen3-vl-embedding (2560维)
- **Rerank**: gte-rerank
- **Vector DB**: Milvus
- **API**: FastAPI

---

## 📦 三种实现方案

### 方案1: 自定义版本 ⭐ 推荐

**位置**: `core/engine.py`

**特点**:
- ✅ 完全控制所有细节
- ✅ 代码清晰易懂
- ✅ 无框架依赖
- ✅ 性能最优

**使用**:
```python
from core import get_engine

engine = get_engine()

# 文搜图
results = await engine.text_search("东京塔夜景", top_k=10)

# 图搜图
results = await engine.image_search("https://.../photo.jpg", top_k=10)

# 图文混合
results = await engine.hybrid_search(
    text="汉服",
    image="https://.../ref.jpg",
    top_k=10
)
```

---

### 方案2: LangChain 版本

**位置**: `core/langchain_engine.py`

**特点**:
- ✅ 生态最丰富
- ✅ 标准化接口
- ✅ 易于集成其他工具
- ⚠️ 需自定义多模态支持

**使用**:
```python
from core.langchain_engine import MultimodalRetriever

retriever = MultimodalRetriever(api_key="sk-xxx")

# 添加数据
retriever.add_travel_memories([...])

# 搜索
results = retriever.search_by_text("东京塔夜景", top_k=10)
```

---

### 方案3: LlamaIndex 版本

**位置**: `core/llamaindex_engine.py`

**特点**:
- ✅ 原生多模态支持
- ✅ 代码最简洁
- ✅ 自动处理文本+图片
- ✅ 专注检索场景

**使用**:
```python
from core.llamaindex_engine import LlamaIndexMultimodalRetriever

retriever = LlamaIndexMultimodalRetriever(api_key="sk-xxx")

# 添加数据 (自动识别图片)
retriever.add_travel_memories([...])

# 搜索
results = retriever.search("东京塔夜景", top_k=10)
```

---

## 📊 方案对比

| 特性 | 自定义 | LangChain | LlamaIndex |
|------|--------|-----------|------------|
| 代码量 | 中等 | 较少 | **最少** |
| 多模态支持 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 生态丰富度 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 学习曲线 | 低 | 中等 | 中等 |
| 生产就绪 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 灵活度 | **高** | 中等 | 中等 |
| 性能 | **最优** | 好 | 好 |

**推荐**:
- 🏆 **快速原型**: 自定义版本
- 🏆 **生产环境**: LangChain 或 LlamaIndex
- 🏆 **多模态RAG**: LlamaIndex (原生支持)
- 🏆 **复杂AI应用**: LangChain (生态最丰富)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# LangChain 版本
pip install langchain langchain-milvus

# LlamaIndex 版本
pip install llama-index llama-index-vector-stores-milvus
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env, 填入 DASHSCOPE_API_KEY
```

### 3. 启动 Milvus

```bash
docker-compose up -d
```

### 4. 运行测试

```bash
# 自定义版本测试
python scripts/test_api.py

# 框架对比测试
python scripts/compare_frameworks.py

# 启动 API
python -m api.main
```

---

## 📡 API 接口

启动后访问: http://localhost:8000/docs

### 核心接口

```bash
# 统一检索
POST /search
{
  "query_text": "东京塔夜景汉服",
  "filters": {"city": "Tokyo"},
  "top_k": 10
}

# 文搜图
POST /search/text?q=东京塔&city=Tokyo&top_k=10

# 图搜图
POST /search/image?image_url=https://...&top_k=10

# 批量入库
POST /ingest
{
  "items": [...]
}
```

---

## 📁 项目结构

```
multimodal-retrieval/
├── api/
│   └── main.py                    # FastAPI 应用
├── core/
│   ├── __init__.py
│   ├── embedder.py                # Qwen3-VL Embedding
│   ├── reranker.py                # gte-rerank
│   ├── milvus_client.py           # Milvus 客户端
│   ├── engine.py                  # 自定义检索引擎 ⭐
│   ├── langchain_engine.py        # LangChain 版本
│   └── llamaindex_engine.py       # LlamaIndex 版本
├── scripts/
│   ├── test_api.py                # API 测试
│   ├── ingest_travel_data.py      # 数据导入
│   └── compare_frameworks.py      # 框架对比
├── tests/
│   └── test_core.py               # 单元测试
├── config/
│   └── settings.py                # 配置管理
├── docker-compose.yml             # Milvus 部署
├── requirements.txt               # 依赖
├── README.md                      # 文档
├── LANGCHAIN_README.md            # LangChain 文档
└── COMPLETE_README.md             # 本文件
```

---

## 🔧 高级功能

### 混合检索 (BM25 + 向量)

```python
from langchain_milvus import BM25BuiltInFunction

vectorstore = Milvus(
    embedding_function=embeddings,
    builtin_functions=[BM25BuiltInFunction()],
    # ...
)
```

### Reranker 集成

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

compressor = CrossEncoderReranker(model="gte-rerank")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 多模态融合编码

```python
# 图文融合为单一向量
embedding = await embedder.embed_fusion(
    text="东京塔夜景",
    images=["https://.../photo1.jpg", "https://.../photo2.jpg"]
)
```

---

## 💰 成本估算

| 项目 | 单价 | 估算用量 | 月费用 |
|-----|------|---------|--------|
| qwen3-vl-embedding | ¥0.258/千图 | 1万张 | ¥2.58 |
| gte-rerank | ¥0.015/千token | 10万次 | ¥15 |
| Milvus | 免费 | - | ¥0 |
| **总计** | | | **~¥18/月** |

---

## 📝 下一步

1. **启动完整服务**
   ```bash
   ./deploy.sh
   ```

2. **导入旅途数据**
   ```bash
   python cli.py ingest --dir /path/to/photos
   ```

3. **测试检索**
   ```bash
   curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query_text": "东京塔夜景汉服"}'
   ```

---

## 📚 参考资料

- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Milvus 文档](https://milvus.io/docs)
- [LangChain 文档](https://python.langchain.com)
- [LlamaIndex 文档](https://docs.llamaindex.ai)

---

_Built with ❤️ by Duckyo 🦊_
# 🦊 Duckyo 多模态检索系统

基于 **Qwen3-VL Embedding + Qwen3-VL Reranker** 的混合向量检索系统，支持文搜图、图搜图、多图搜图、图文混合搜索。

## ✨ 特性

- **多模态 Embedding**: Qwen3-VL 原生支持文本、图片、视频
- **融合模式**: 支持图文混合编码为单一向量
- **两阶段检索**: 向量粗筛 + 多模态精排
- **Milvus 向量库**: 高性能 ANN 索引
- **FastAPI**: 友好的 RESTful API
- **CLI 工具**: 命令行操作更便捷

## 🚀 快速开始

### 方式一: 一键部署 (推荐)

```bash
cd /root/.openclaw/workspace/duckyo/projects/multimodal-retrieval
./deploy.sh
```

### 方式二: 手动部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env, 填入 DASHSCOPE_API_KEY

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动 Milvus
docker-compose up -d

# 4. 初始化 Collection
python cli.py init

# 5. 启动 API
python -m api.main
```

### 方式三: Docker 部署

```bash
docker-compose up -d
```

## 📡 API 接口

启动后访问: http://localhost:8000/docs

### 统一检索接口

```bash
POST /search
```

**请求示例:**

```json
{
  "query_text": "东京塔夜景汉服",
  "filters": {"city": "Tokyo"},
  "top_k": 10,
  "use_rerank": true
}
```

### 文搜图

```bash
POST /search/text?q=东京塔夜景&city=Tokyo&top_k=10
```

### 图搜图

```bash
POST /search/image?image_url=https://example.com/photo.jpg&top_k=10
```

### 多图搜图

```bash
POST /search/images?image_urls=["url1","url2"]&top_k=10
```

### 批量入库

```bash
POST /ingest
{
  "items": [
    {
      "id": "photo_001",
      "type": "image",
      "content": "东京塔夜景汉服照",
      "file_url": "https://example.com/tokyo.jpg",
      "city": "Tokyo",
      "country": "Japan",
      "date": "2026-03-08",
      "tags": ["汉服", "夜景"],
      "mood": "开心"
    }
  ]
}
```

## 🖥️ CLI 工具

```bash
# 初始化数据库
python cli.py init

# 文搜图
python cli.py search "东京塔夜景" --city Tokyo

# 图搜图
python cli.py search-image https://example.com/photo.jpg

# 导入照片
python cli.py ingest --dir /path/to/photos

# 查看统计
python cli.py stats
```

## 🐳 Docker 服务

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f api

# 停止服务
docker-compose down
```

服务端口:
- API: `http://localhost:8000`
- API 文档: `http://localhost:8000/docs`
- Milvus: `http://localhost:19530`
- Attu (管理界面): `http://localhost:3000`

## 📁 项目结构

```
multimodal-retrieval/
├── api/
│   └── main.py              # FastAPI 应用
├── cli.py                   # CLI 工具
├── config/
│   └── settings.py          # 配置管理
├── core/
│   ├── embedder.py          # Qwen3-VL Embedding
│   ├── reranker.py          # Qwen3-VL Reranker
│   ├── milvus_client.py     # Milvus 客户端
│   └── engine.py            # 检索引擎
├── scripts/
│   ├── test_api.py          # API 测试
│   └── ingest_travel_data.py # 批量导入
├── tests/
│   └── test_core.py         # 单元测试
├── deploy.sh                # 一键部署脚本
├── start.sh                 # 启动脚本
├── docker-compose.yml       # Docker 编排
├── Dockerfile               # API 镜像
├── requirements.txt         # Python 依赖
├── .env.example            # 环境变量模板
└── README.md               # 使用文档
```

## 💰 成本估算

| 项目 | 单价 | 月用量 | 月费用 |
|-----|------|--------|--------|
| Qwen3-VL Embedding | ¥0.258/千图 | 1万张 | ¥2.58 |
| Qwen3-VL Reranker | ¥0.015/千token | 10万次 | ¥15 |
| **合计** | | | **~¥18/月** |

## 📝 示例代码

### Python 使用

```python
from core import get_engine

engine = get_engine()

# 文搜图
results = await engine.text_search(
    query="东京塔夜景",
    filters={"city": "Tokyo"},
    top_k=10
)

# 图搜图
results = await engine.image_search(
    image_url="https://example.com/photo.jpg",
    top_k=10
)

# 多图搜图
results = await engine.multi_image_search(
    image_urls=["url1", "url2", "url3"],
    top_k=10
)

# 图文混合
results = await engine.hybrid_search(
    text="汉服",
    image="https://example.com/ref.jpg",
    top_k=10
)
```

### cURL 使用

```bash
# 文搜图
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "东京塔夜景汉服",
    "top_k": 10
  }'

# 图搜图
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_image": "https://example.com/photo.jpg",
    "top_k": 10
  }'
```

## 🔧 常用命令

```bash
# 一键部署
./deploy.sh

# 启动服务
./start.sh

# CLI 工具
python cli.py --help

# 运行测试
python scripts/test_api.py
pytest tests/

# 查看日志
docker-compose logs -f

# 进入 Milvus 容器
docker-compose exec milvus-standalone bash
```

## 📚 常见问题

### Q: 如何处理图片 URL 是本地文件的情况?

A: 需要先将本地图片上传到可访问的 URL (如 MinIO、OSS、Imgur 等)

### Q: Reranker 是否必须使用?

A: 可以选择。设置 `use_rerank: false` 可以跳过精排阶段，减少 API 调用成本

### Q: 如何批量导入已有照片?

A: 使用 `python cli.py ingest --dir /path/to/photos` 或调用 `/ingest` 接口

### Q: 如何查看 Collection 统计信息?

A: 访问 `http://localhost:8000/collection/stats` 或使用 `python cli.py stats`

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_core.py -v

# 测试 API
python scripts/test_api.py
```

## 📜 License

MIT License

---

_Built with ❤️ by Duckyo 🦊_
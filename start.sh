#!/bin/bash
# 启动多模态检索引擎 (Milvus + MinIO)

set -e

echo "🦊 Duckyo 多模态检索引擎启动脚本"
echo "=================================="

# 检查 Docker
echo "🔍 检查 Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose 未安装"
    exit 1
fi

# 启动 Milvus 核心服务
echo "🚀 启动 Milvus 核心服务..."
docker compose up -d milvus-etcd milvus-minio milvus-standalone

# 等待 Milvus 就绪
echo "⏳ 等待 Milvus 就绪..."
sleep 10

# 检查 Milvus 健康
until curl -s http://localhost:9091/healthz > /dev/null 2>&1; do
    echo "  等待 Milvus 健康检查..."
    sleep 5
done

echo "✅ Milvus 已就绪"
echo ""
echo "=================================="
echo "✅ 服务已启动!"
echo ""
echo "🔍 Milvus: localhost:19530"
echo "🖥️  MinIO Console: localhost:9003"
echo "=================================="
echo ""
echo "📝 使用方式:"
echo "   python -c \"import asyncio; from core import get_engine; e = get_engine(); asyncio.run(e.text_search('纽约'))\""
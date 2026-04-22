#!/bin/bash
# LlamaIndex 多模态检索系统 - 快速启动

echo "🦊 LlamaIndex 多模态检索系统 - 快速启动"
echo "=========================================="

# 检查 Milvus
echo ""
echo "📦 检查 Milvus 服务..."
if ! curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
    echo "⚠️  Milvus 未运行，正在启动..."
    docker-compose up -d
    sleep 15
fi

echo "✅ Milvus 服务正常"

# 启动 API
echo ""
echo "🚀 启动 LlamaIndex API 服务..."
echo ""
echo "=========================================="
echo "📡 API 地址：http://localhost:8001"
echo "📖 API 文档：http://localhost:8001/docs"
echo "=========================================="
echo ""

python3 api/llamaindex_api.py
#!/bin/bash
# Start Multimodal Retrieval Engine (Milvus + MinIO)

set -e

echo "[INFO] Duckyo Multimodal Retrieval Engine Startup Script"
echo "=================================="

# Check Docker
echo "[INFO] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "[ERROR] Docker Compose not installed"
    exit 1
fi

# Start Milvus core services
echo "[INFO] Starting Milvus core services..."
docker compose up -d milvus-etcd milvus-minio milvus-standalone

# Wait for Milvus ready
echo "[INFO] Waiting for Milvus ready..."
sleep 10

# Check Milvus health
until curl -s http://localhost:9091/healthz > /dev/null 2>&1; do
    echo "  Waiting for Milvus health check..."
    sleep 5
done

echo "[OK] Milvus is ready"
echo ""
echo "=================================="
echo "[OK] Services started!"
echo ""
echo "[INFO] Milvus: localhost:19530"
echo "[INFO] MinIO Console: localhost:9003"
echo "=================================="
echo ""
echo "[INFO] Usage:"
echo "   python -c \"import asyncio; from core import get_engine; e = get_engine(); asyncio.run(e.text_search('New York'))\""

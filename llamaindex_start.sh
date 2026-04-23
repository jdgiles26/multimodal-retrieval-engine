#!/bin/bash
# LlamaIndex Multimodal Retrieval System - Quick Start

echo "[INFO] LlamaIndex Multimodal Retrieval System - Quick Start"
echo "=========================================="

# Check Milvus
echo ""
echo "[INFO] Checking Milvus service..."
if ! curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
    echo "[WARN] Milvus not running, starting..."
    docker-compose up -d
    sleep 15
fi

echo "[OK] Milvus service normal"

# Start API
echo ""
echo "[INFO] Starting LlamaIndex API service..."
echo ""
echo "=========================================="
echo "[INFO] API URL: http://localhost:8001"
echo "[INFO] API Docs: http://localhost:8001/docs"
echo "=========================================="
echo ""

python3 api/llamaindex_api.py

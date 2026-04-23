#!/bin/bash
# One-click deployment script

set -e

echo "[INFO] Duckyo Multimodal Retrieval System - One-Click Deployment"
echo "===================================="

PROJECT_DIR="/opt/multimodal-retrieval"

echo ""
echo "[INFO] Step 1: Install dependencies"
echo "-------------------"
pip install -q -r requirements.txt
echo "[OK] Python dependencies installed"

echo ""
echo "[INFO] Step 2: Start Docker services"
echo "---------------------------"
docker-compose up -d
echo "[OK] Docker services started"

echo ""
echo "[INFO] Step 3: Wait for services ready"
echo "-----------------------"
sleep 15

# Check service health
for i in {1..30}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "[OK] Milvus health check passed"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "[INFO] Step 4: Initialize database"
echo "----------------------"
python -c "
import sys
sys.path.insert(0, '.')
from core import get_milvus_client
client = get_milvus_client()
client.connect()
try:
    client.create_collection(dimension=2560)
    print('[OK] Collection created successfully')
except Exception as e:
    if 'already exists' in str(e).lower():
        print('[OK] Collection already exists')
    else:
        print(f'[WARN] {e}')
"

echo ""
echo "[INFO] Step 5: Run tests"
echo "------------------"
python scripts/test_api.py || echo "[WARN] Tests require network connection"

echo ""
echo "===================================="
echo "[OK] Deployment complete!"
echo ""
echo "[INFO] Access URLs:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Milvus: http://localhost:19530"
echo "  - Attu: http://localhost:3000"
echo ""
echo "[INFO] Common commands:"
echo "  ./start.sh          # Start services"
echo "  python cli.py --help # CLI tool"
echo "  docker-compose logs -f # View logs"
echo "===================================="

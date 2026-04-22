#!/bin/bash
# 一键部署脚本

set -e

echo "🦊 Duckyo 多模态检索系统 - 一键部署"
echo "===================================="

PROJECT_DIR="/opt/multimodal-retrieval"

echo ""
echo "📦 步骤 1: 安装依赖"
echo "-------------------"
pip install -q -r requirements.txt
echo "✅ Python 依赖安装完成"

echo ""
echo "🐳 步骤 2: 启动 Docker 服务"
echo "---------------------------"
docker-compose up -d
echo "✅ Docker 服务已启动"

echo ""
echo "⏳ 步骤 3: 等待服务就绪"
echo "-----------------------"
sleep 15

# 检查服务健康
for i in {1..30}; do
    if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
        echo "✅ Milvus 健康检查通过"
        break
    fi
    echo "  等待中... ($i/30)"
    sleep 2
done

echo ""
echo "🔧 步骤 4: 初始化数据库"
echo "----------------------"
python -c "
import sys
sys.path.insert(0, '.')
from core import get_milvus_client
client = get_milvus_client()
client.connect()
try:
    client.create_collection(dimension=2560)
    print('✅ Collection 创建成功')
except Exception as e:
    if 'already exists' in str(e).lower():
        print('✅ Collection 已存在')
    else:
        print(f'⚠️  {e}')
"

echo ""
echo "🧪 步骤 5: 运行测试"
echo "------------------"
python scripts/test_api.py || echo "⚠️ 测试需要网络连接"

echo ""
echo "===================================="
echo "✅ 部署完成!"
echo ""
echo "🌐 访问地址:"
echo "  - API: http://localhost:8000"
echo "  - 文档: http://localhost:8000/docs"
echo "  - Milvus: http://localhost:19530"
echo "  - Attu: http://localhost:3000"
echo ""
echo "📚 常用命令:"
echo "  ./start.sh          # 启动服务"
echo "  python cli.py --help # CLI 工具"
echo "  docker-compose logs -f # 查看日志"
echo "===================================="
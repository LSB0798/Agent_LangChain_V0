#!/bin/bash
echo "开始清理 Milvus 环境..."

# 杀死所有 Milvus 相关进程
pkill -f milvus 2>/dev/null || echo "没有找到 Milvus 进程"

# 杀死占用端口的进程
for port in 19530 19531 19532 19533; do
    if lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null ; then
        echo "杀死占用端口 ${port} 的进程..."
        kill -9 $(lsof -t -i:${port}) 2>/dev/null
    fi
done

# 清理 Docker 容器（如果有）
docker stop $(docker ps -aq) 2>/dev/null || echo "没有运行的 Docker 容器"
docker rm $(docker ps -aq) 2>/dev/null || echo "没有 Docker 容器可删除"

# 确保数据目录存在且有正确权限
mkdir -p /data/lishuaibing/milvus_data
chmod 777 /data/lishuaibing/milvus_data

echo "清理完成"

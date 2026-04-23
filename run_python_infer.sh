#!/bin/bash
# file: run_all.sh

# 把顺序执行体放到后台
nohup bash -c '
for py in ./langchain_qwen3_Milvus_22_new.py
do
    log=${py%.*}.log          # 去掉 .py 加 .log
    echo "[$(date)] 开始运行 $py ，日志：$log"
    python -u "$py" > "$log" 2>&1
    if [ $? -eq 0 ]; then
        echo "[$(date)] $py 运行完成"
    else
        echo "[$(date)] $py 运行失败，详见 $log"
    fi
done
' > run_all.out 2>&1 &

echo "全部任务已送入后台，顺序执行中... 可随时查看 run_all.out 了解进度。"

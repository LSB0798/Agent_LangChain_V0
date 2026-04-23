"""
RAG API 服务（基于 Milvus + Embedding + Reranker）
提供异步并发查询接口，参考 langchain_qwen3_Milvus_22_new.py 的 API 风格
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# 导入原有 RAG 系统（需确保路径正确）
# 假设 langchain_qwen3_Milvus_26_test_milvus.py 在当前目录或已加入 sys.path
from langchain_qwen3_Milvus_26_test_milvus import EnhancedRAGSystem, Config

# ==================== 配置 ====================
class APIConfig:
    HOST = "0.0.0.0"
    PORT = 8001
    # 并发限制（同时处理的最大请求数，超出排队）
    MAX_CONCURRENT_REQUESTS = 100
    # 请求超时（秒）
    REQUEST_TIMEOUT = 60

# ==================== 请求/响应模型 ====================
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询文本")
    use_memory: bool = Field(True, description="是否使用记忆系统")

class QueryResponse(BaseModel):
    status: str
    query: str
    answer: str
    elapsed_time: float  # 处理耗时（秒）
    timestamp: str

# 兼容 OpenAI 风格（可选）
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "rag-model"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 2000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# ==================== 全局单例 ====================
_rag_system: Optional[EnhancedRAGSystem] = None

def get_rag_system() -> EnhancedRAGSystem:
    """延迟初始化 RAG 系统（线程安全）"""
    global _rag_system
    if _rag_system is None:
        print("正在初始化 RAG 系统（模型加载、Milvus 连接等）...")
        start = time.time()
        config = Config()
        _rag_system = EnhancedRAGSystem(config)
        print(f"RAG 系统初始化完成，耗时 {time.time()-start:.2f} 秒")
    return _rag_system

# ==================== FastAPI 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：预热系统（可选）
    print("启动 RAG API 服务...")
    # 在后台线程中初始化，避免阻塞事件循环
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_rag_system)
    yield
    # 关闭时：清理资源
    if _rag_system:
        print("正在清理 RAG 系统资源...")
        await loop.run_in_executor(None, _rag_system.cleanup)
        print("清理完成")

app = FastAPI(
    title="RAG 问答系统 API",
    description="基于 Milvus + Qwen3 Embedding + Reranker 的检索增强生成",
    version="1.0",
    lifespan=lifespan
)

# ==================== 辅助函数（同步转异步） ====================
async def run_rag_query(query: str, use_memory: bool) -> str:
    """
    在 executor 中执行同步的 answer_query，避免阻塞事件循环
    """
    rag = get_rag_system()
    loop = asyncio.get_event_loop()
    # 注意：answer_query 内部可能耗时较长（检索+重排序），使用 to_thread
    result = await loop.run_in_executor(
        None,  # 使用默认线程池
        lambda: rag.answer_query(query, use_memory=use_memory)
    )
    return result

# ==================== 自定义端点 ====================
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    start = time.time()
    try:
        answer = await asyncio.wait_for(run_rag_query(req.query, req.use_memory), timeout=APIConfig.REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(504, "请求处理超时")
    except Exception as e:
        import traceback
        traceback.print_exc()   # 打印到服务端控制台
        raise HTTPException(500, f"处理失败: {str(e)}")
    return QueryResponse(status="success", query=req.query, answer=answer, elapsed_time=time.time()-start, timestamp=datetime.now().isoformat())

# ==================== OpenAI 兼容端点（可选） ====================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    兼容 OpenAI 风格的接口，从 messages 中提取最后一条 user 消息作为查询
    """
    # 提取用户输入
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break
    if not user_query:
        raise HTTPException(status_code=400, detail="未找到用户消息")
    
    start_time = time.time()
    try:
        answer = await asyncio.wait_for(
            run_rag_query(user_query, use_memory=True),
            timeout=APIConfig.REQUEST_TIMEOUT
        )
        elapsed = time.time() - start_time
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # 构造 OpenAI 格式响应
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_query),
            "completion_tokens": len(answer),
            "total_tokens": len(user_query) + len(answer)
        }
    }

# ==================== 健康检查 ====================
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ==================== 启动入口 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_api:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        workers=2,                     # 多进程 worker，充分利用多核
        limit_concurrency=APIConfig.MAX_CONCURRENT_REQUESTS,  # 最大并发连接数
        backlog=2048,                  # 监听队列
        timeout_keep_alive=30,
        log_level="info"
    )
"""
base on langchain_qwen3_Milvus_15_new.py
1, ReAct 代理的决策逻辑​
    ReAct 代理会根据问题类型自动选择工具，优先级通常是：
        ​简单问题​ → direct_response（直接回答）
        ​需要知识​ → document_search（文档搜索）
        ​需要上下文​ → memory_retrieval（记忆检索）
        ​复杂/模糊问题​ → prompt_optimization（提示优化）
2, 
3, 
4, 修改了prompts，减轻了对回复的约束
5, 
6, _react_prompt_optimization
7, 将Milvus对RAG知识库的管理纳入到ReAct中，作为一个工具，工具路径是：
    7-1: _fallback_answer_query 中初始检索+重排序;
    7-2: def _react_document_search -> def _init_tools -> self.tools -> agent = create_react_agent
    7-3: 完整的调用链
            EnhancedRAGSystem.__init__() 
                ↓
            self.tools = self._init_tools()
                ↓
            def _init_tools():
                ↓
            Tool(name="document_search", func=self._react_document_search)
                ↓
            ReAct 代理执行时根据需要调用 document_search 工具
                ↓
            实际执行 self._react_document_search(query)
8，将Milvus对Memory的管理纳入到ReAct中，作为一个工具，工具路径是：
    8-1: 可以检索之前的问题和回复，比如用history;
    8-2: 结束对话保留聊天记录 Memory; 
    8-3: 对memory进行reranker管理;
    8-4: 可以清空聊天记录 clear;
    8-5: 可以debug_memory;
    8-6: 检索和重排序在 def retrieve_memories_with_reranker 当中;
    8-7: 主流程调用路径​：
            EnhancedRAGSystem.answer_query()
                ↓
            EnhancedRAGSystem._fallback_answer_query()
                ↓
            AgentMemorySystem.get_contextual_prompt()
                ↓
            AgentMemorySystem.retrieve_memories_with_reranker()
                ↓
            AgentMemorySystem.retrieve_memories()  # 向量检索
                ↓
            AgentMemorySystem._rerank_memories()   # reranker重排序
    8-8: ReAct 代理调用路径​：def retrieve_memories_with_reranker -> def _react_memory_retrieval -> def _init_tools -> self.tools -> agent = create_react_agent
            EnhancedRAGSystem.answer_query()
                ↓
            ReAct Agent 思考过程
                ↓
            调用 "memory_retrieval" 工具
                ↓
            EnhancedRAGSystem._react_memory_retrieval()
                ↓
            AgentMemorySystem.retrieve_memories_with_reranker()
    8-9: 影响 memory 检索行为的配置：
            class Config:
                INITIAL_RETRIEVAL_K = 20           # 初始检索数量
                RERANKER_TOP_K = 5                 # 重排序后返回数量
                MEMORY_RETRIEVAL_WEIGHT = 0.3      # 记忆检索权重
                MAX_CONVERSATION_TURNS = 10        # 最大对话轮次记忆
8, 创建带调试功能的工具包装器，def create_debug_tool(name, func, description) 对调用工具进行监控
"""

"""
相比于 langchain_qwen3_Milvus_16_new.py：
1. ​ReAct代理执行逻辑的重大改进​
​文档2在answer_query方法中进行了关键优化：

添加了return_intermediate_steps=True参数来获取中间步骤
新增了_build_thinking_from_intermediate_steps方法，从中间步骤构建更可靠的思考过程
改进了最终答案检测逻辑_has_complete_final_answer
2. ​最终答案提取逻辑增强​
​文档2改进了_extract_final_answer方法：

使用正则表达式进行更精确的模式匹配
支持多种最终答案标记格式（中文/英文）
增加了内容长度检查，确保答案有实质性内容
3. ​思考过程构建优化​
​文档2新增了完整的中间步骤处理：

python
运行
复制
def _build_thinking_from_intermediate_steps(self, intermediate_steps):
    # 从代理执行的每个步骤中提取思考、行动和观察
    # 比文档1仅依赖最终输出更可靠
4. ​提示词构建逻辑改进​
​文档2的_build_enhanced_prompt方法：

根据思考过程的质量动态调整提示词
当思考过程有价值时使用详细提示，否则使用简化提示
增加了长度检查避免使用无意义的思考过程
5. ​错误处理和调试增强​
​文档2增加了更多的调试输出：

打印中间步骤数量和信息
显示构建的思考过程内容
增强的错误处理和异常捕获
"""

import re
import os
import torch
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer as RerankerTokenizer

# 引入 Milvus Lite 相关组件
from milvus import default_server
from langchain_community.vectorstores import Milvus
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, connections, utility

from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader

from openai import OpenAI

class JSONLLoader(BaseLoader):
    """自定义 JSONL 文件加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载并解析 JSONL 文件"""
        documents = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # 解析每行的 JSON 对象
                        data = json.loads(line)
                        # 将整个 JSON 对象转换为字符串作为内容
                        content = json.dumps(data, ensure_ascii=False, indent=2)
                        documents.append(Document(
                            page_content=content,
                            metadata={
                                "source": self.file_path,
                                "line_number": line_num,
                                "type": "jsonl"
                            }
                        ))
                    except json.JSONDecodeError as e:
                        print(f"解析 JSONL 文件错误（行 {line_num}）: {e}")
        except Exception as e:
            print(f"读取 JSONL 文件失败: {e}")
        return documents

class Config:
    # 文档相关配置
    DOCUMENTS_DIR = "documents"  # 本地文档目录（需手动创建）
    CHUNK_SIZE = 1000             # 每个文本块的字符数（中文适配）
    CHUNK_OVERLAP = 50           # 块间重叠字符数（避免分割丢失上下文）
    
    # 模型相关配置
    EMBEDDING_MODEL_NAME = "/data/lishuaibing/Qwens/Qwen3-Embedding-0.6B"  # 使用Qwen3嵌入模型
    RERANKER_MODEL_NAME = "/data/lishuaibing/Qwens/Qwen3-Reranker-0.6B"    # Reranker模型
    LLM_MODEL_NAME = "/data/lishuaibing/Qwen3-30B-A3B"  # LLM模型

    # 检索相关配置
    COLLECTION_NAME = "rag_collection"  # 集合名称
    MEMORY_COLLECTION_NAME = "memory_collection"  # 记忆集合名称
    INITIAL_RETRIEVAL_K = 20           # 初始检索数量
    RERANKER_TOP_K = 5                 # 重排序后返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）
    
    # Memory相关配置
    MAX_CONVERSATION_TURNS = 10  # 最大对话轮次记忆
    MAX_SUMMARY_LENGTH = 500     # 摘要最大长度
    MEMORY_RETRIEVAL_WEIGHT = 0.3  # 记忆检索权重
    ENABLE_MEMORY_SUMMARY = True   # 启用记忆摘要

    client = OpenAI(base_url="http://10.20.223.89:61253/v1", api_key='EMPTY')

    # 新增语言检测配置
    LANG_DETECTION_THRESHOLD = 0.5  # 语言检测置信度阈值
    ENGLISH_CHUNK_SIZE = 1500       # 英文为主文档的块大小
    ENGLISH_CHUNK_OVERLAP = 80     # 英文为主文档的块重叠
    MIXED_CHUNK_SIZE = 1200         # 混合文档的块大小
    MIXED_CHUNK_OVERLAP = 65       # 混合文档的块重叠

    # 新增记忆集合配置
    PROCEDURAL_MEMORY_COLLECTION = "procedural_memory"  # 长期偏好记忆
    EPISODIC_MEMORY_COLLECTION = "episodic_memory"      # 情景记忆（对话历史）
    SEMANTIC_MEMORY_COLLECTION = "semantic_memory"      # 语义记忆（事实知识）

    # 新增 ReAct 代理配置
    REACT_LLM_MODEL_NAME = "/data/lishuaibing/qwen3/Qwen3-0___6B/"  # ReAct 使用的小模型
    REACT_MAX_NEW_TOKENS = 300  # ReAct 推理的 token 限制
    REACT_TEMPERATURE = 0.5     # ReAct 推理的温度

def detect_language(text: str) -> Dict[str, float]:
    """检测文本语言类型及比例（简单实现）"""
    import re
    # 统计中英文字符比例
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    total = max(chinese_chars + english_chars, 1)
    
    return {
        "is_english": english_chars / total > Config.LANG_DETECTION_THRESHOLD,
        "is_mixed": 0.3 < english_chars / total < 0.7,  # 30%-70%视为混合
        "english_ratio": english_chars / total
    }

# 替换原有的ConversationMemory类
class AgentMemorySystem:
    """基于Milvus的多类型记忆管理系统（完全移除JSON依赖）"""
    
    def __init__(self, config, embeddings, reranker_model=None, reranker_tokenizer=None):
        self.config = config
        self.embeddings = embeddings
        self.reranker_model = reranker_model
        self.reranker_tokenizer = reranker_tokenizer
        self._check_embedding_dimension()
        self._init_memory_collections()
        print("记忆系统初始化完成（支持reranker优化）")
    
    def _format_memory_instruction(self, query, memory_content):
        """格式化记忆检索的指令"""
        return f"<Instruct>: Given a user query, retrieve relevant memories that help understand the user's context and preferences.\n<Query>: {query}\n<Document>: {memory_content}"

    def _rerank_memories(self, query, memories, top_k=3):
        """使用reranker对记忆进行重排序"""
        if not memories or not self.reranker_model:
            return memories[:top_k]
            
        try:
            # 准备重排序对
            pairs = [self._format_memory_instruction(query, mem["content"]) for mem in memories]
            
            # 处理输入
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            device = next(self.reranker_model.parameters()).device
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            
            # 计算相关性分数
            batch_scores = self.reranker_model(**inputs).logits[:, -1, :]
            token_false_id = self.reranker_tokenizer.convert_tokens_to_ids('no')
            token_true_id = self.reranker_tokenizer.convert_tokens_to_ids('yes')
            
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            
            # 组合记忆和分数
            memory_scores = list(zip(memories, scores))
            memory_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回重排序后的记忆
            return [mem for mem, score in memory_scores[:top_k]]
            
        except Exception as e:
            print(f"记忆重排序失败: {e}")
            return memories[:top_k]

    def retrieve_memories_with_reranker(self, query, memory_type="all", top_k=5, min_importance=0.3, user_id="default"):
        """使用reranker优化的记忆检索"""
        try:
            # 第一步：向量检索（获取更多候选）
            vector_results = self.retrieve_memories(
                query, 
                memory_type=memory_type, 
                top_k=top_k * 3,  # 获取更多候选用于重排序
                min_importance=min_importance, 
                user_id=user_id
            )
            
            # 第二步：使用reranker重排序
            reranked_results = {}
            
            for mem_type, memories in vector_results.items():
                if memories:
                    reranked_memories = self._rerank_memories(query, memories, top_k=top_k)
                    reranked_results[mem_type] = reranked_memories
                else:
                    reranked_results[mem_type] = []
            
            return reranked_results
            
        except Exception as e:
            print(f"reranker记忆检索失败: {e}")
            return self.retrieve_memories(query, memory_type, top_k, min_importance, user_id)
    
    def _check_embedding_dimension(self):
        """检查嵌入模型的真实维度"""
        try:
            # 测试嵌入向量的维度
            test_embedding = self.embeddings.embed_query("测试文本")
            self.actual_embedding_dim = len(test_embedding)
            print(f"嵌入模型实际维度: {self.actual_embedding_dim}")
        except Exception as e:
            print(f"检查嵌入维度失败: {e}")
            # 根据错误信息，嵌入模型输出1024维
            self.actual_embedding_dim = 1024
            print(f"使用修复后的维度: {self.actual_embedding_dim}")
    
    def _init_memory_collections(self):
        """初始化三种记忆集合"""
        try:
            # 检查并创建记忆集合
            self.procedural_memory = self._get_or_create_collection(
                self.config.PROCEDURAL_MEMORY_COLLECTION, 
                "用户偏好与行为规则"
            )
            self.episodic_memory = self._get_or_create_collection(
                self.config.EPISODIC_MEMORY_COLLECTION,
                "对话历史与事件记录"
            )
            self.semantic_memory = self._get_or_create_collection(
                self.config.SEMANTIC_MEMORY_COLLECTION,
                "事实性知识"
            )
            print("记忆集合初始化完成")
        except Exception as e:
            print(f"初始化记忆集合失败: {e}")
    
    def _check_dimension_match(self, collection):
        """检查现有集合的维度是否与当前模型匹配"""
        try:
            # 获取集合的schema信息
            schema = collection.schema
            for field in schema.fields:
                if field.name == "embedding" and field.dtype == DataType.FLOAT_VECTOR:
                    # 获取嵌入字段的维度
                    existing_dim = field.params.get("dim")
                    if existing_dim == self.actual_embedding_dim:
                        print(f"维度匹配: 现有集合维度 {existing_dim}，当前模型维度 {self.actual_embedding_dim}")
                        return True
                    else:
                        print(f"维度不匹配: 现有集合维度 {existing_dim}，当前模型维度 {self.actual_embedding_dim}")
                        return False
            # 如果没有找到embedding字段，返回False
            print("未找到embedding字段")
            return False
        except Exception as e:
            print(f"检查维度匹配时出错: {e}")
            return False
    
    def _get_or_create_collection(self, name, description):
        """获取或创建记忆集合（修复维度问题），启动时候清理掉旧的 memory"""
        
        """# 先删除可能存在的旧集合（维度不匹配）
        try:
            if utility.has_collection(name):
                utility.drop_collection(name)
                print(f"删除旧集合 {name}（维度不匹配）")
        except Exception as e:
            print(f"删除旧集合时出错: {e}")"""
        
        # 检查集合是否存在且维度匹配
        if utility.has_collection(name):
            try:
                # 获取现有集合的维度信息
                collection = Collection(name)
                # 这里需要获取集合的schema来检查维度
                # 如果维度匹配，直接使用现有集合
                if self._check_dimension_match(collection):
                    collection.load()
                    print(f"使用现有记忆集合: {name}")
                    return collection
                else:
                    # 只有维度不匹配时才清理
                    utility.drop_collection(name)
                    print(f"删除旧集合 {name}（维度不匹配）")
            except Exception as e:
                print(f"检查现有集合时出错: {e}")
                utility.drop_collection(name)
        
        # 创建新的记忆集合，使用动态维度
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.actual_embedding_dim),
            FieldSchema(name="importance", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        
        schema = CollectionSchema(fields=fields, description=description)
        collection = Collection(name=name, schema=schema)
        
        # 创建索引
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.create_index(field_name="user_id", index_params={"index_type": "TRIE"})
        collection.load()
        
        print(f"创建新记忆集合: {name}, 维度: {self.actual_embedding_dim}")
        return collection

    def store_memory(self, memory_type, content, importance=0.5, metadata=None, user_id="default"):
        """存储记忆到指定类型集合（修复维度问题）"""
        try:
            if memory_type == "procedural":
                collection = self.procedural_memory
            elif memory_type == "episodic":
                collection = self.episodic_memory
            elif memory_type == "semantic":
                collection = self.semantic_memory
            else:
                raise ValueError(f"未知的记忆类型: {memory_type}")
            
            # 检查内容长度，如果超过限制则截断
            MAX_CONTENT_LENGTH = 19000  # 留一些余量
            if len(content) > MAX_CONTENT_LENGTH:
                print(f"警告: 记忆内容长度 {len(content)} 超过限制，将被截断到 {MAX_CONTENT_LENGTH}")
                content = content[:MAX_CONTENT_LENGTH]
            
            # 生成嵌入向量
            embedding = self.embeddings.embed_query(content)
            
            # 验证维度
            if len(embedding) != self.actual_embedding_dim:
                print(f"警告: 嵌入维度不匹配，期望{self.actual_embedding_dim}，实际{len(embedding)}")
                # 如果维度不匹配，截断或填充到正确维度
                if len(embedding) > self.actual_embedding_dim:
                    embedding = embedding[:self.actual_embedding_dim]
                else:
                    # 填充到正确维度
                    embedding.extend([0.0] * (self.actual_embedding_dim - len(embedding)))
            
            # 准备数据
            data = [{
                "user_id": user_id,
                "content": content,
                "embedding": embedding,
                "importance": importance,
                "created_at": int(datetime.now().timestamp()),
                "metadata": metadata or {}
            }]
            
            # 插入数据
            collection.insert(data)
            collection.flush()
            
            print(f"✅ 已存储{memory_type}记忆: {content[:50]}...")
            return True
            
        except Exception as e:
            print(f"存储记忆失败: {e}")
            return False

    def retrieve_memories(self, query, memory_type="all", top_k=5, min_importance=0.3, user_id="default"):
        """检索相关记忆"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = {}
            
            # 构建过滤表达式
            filter_expr = f'user_id == "{user_id}" && importance >= {min_importance}'
            search_params = {"metric_type": "IP", "params": {"ef": 100}}
            
            # 确定要检索的集合
            collections_to_search = []
            if memory_type in ["all", "procedural"]:
                collections_to_search.append(("procedural", self.procedural_memory))
            if memory_type in ["all", "episodic"]:
                collections_to_search.append(("episodic", self.episodic_memory))
            if memory_type in ["all", "semantic"]:
                collections_to_search.append(("semantic", self.semantic_memory))
            
            # 执行检索
            for mem_type, collection in collections_to_search:
                search_results = collection.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=filter_expr,
                    output_fields=["content", "importance", "created_at", "metadata"]
                )
                if search_results:
                    results[mem_type] = []
                    for hit in search_results[0]:
                        results[mem_type].append({
                            "content": hit.entity.get("content"),
                            "importance": hit.entity.get("importance"),
                            "score": hit.score,
                            "metadata": hit.entity.get("metadata", {})
                        })
            
            return results
            
        except Exception as e:
            print(f"检索记忆失败: {e}")
            return {}

    def add_conversation_turn(self, query: str, response: str, context: List[str] = None):
        """添加对话轮次到记忆系统（增强版）"""
        try:
            # 简化存储逻辑，只存储完整的对话轮次，避免重复存储导致的问题
            self.store_memory(
                memory_type="episodic",
                content=f"用户问题: {query}\n助手回答: {response}",
                importance=0.8,
                metadata={
                    "context": context or [],
                    "type": "conversation_turn",
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "response": response,
                    "query_length": len(query),
                    "response_length": len(response)
                }
            )
            print(f"✅ 已存储对话轮次: Q: {query[:50]}... A: {response[:50]}...")
            
        except Exception as e:
            print(f"存储对话轮次失败: {e}")
    
    def get_relevant_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """获取与当前查询相关的历史记忆"""
        # 从情景记忆中检索相关对话
        episodic_memories = self.retrieve_memories(
            query, 
            memory_type="episodic", 
            top_k=top_k,
            min_importance=0.3
        )
        
        relevant_memories = []
        if "episodic" in episodic_memories:
            for mem in episodic_memories["episodic"]:
                # 解析对话记忆内容
                if "用户:" in mem["content"] and "助手:" in mem["content"]:
                    parts = mem["content"].split("\n")
                    if len(parts) >= 2:
                        relevant_memories.append({
                            "query": parts[0].replace("用户: ", ""),
                            "response": parts[1].replace("助手: ", ""),
                            "score": mem["score"],
                            "timestamp": mem["metadata"].get("timestamp", "")
                        })
        
        return relevant_memories[:top_k]

    def get_recent_conversation_history(self, limit=10):
        """获取最近的对话历史（增强版）"""
        try:
            # 使用更精确的查询条件
            results = self.episodic_memory.query(
                expr='metadata["type"] == "conversation_turn"',
                output_fields=["content", "metadata", "created_at"],
                limit=limit * 2,  # 多取一些，确保有足够的数据
                order_by="created_at desc"
            )
            
            # 格式化结果
            conversation_history = []
            for result in results:
                metadata = result.get("metadata", {})
                
                # 优先从metadata中获取，这是最可靠的
                if "query" in metadata and "response" in metadata:
                    # 验证数据完整性
                    if (metadata.get("query_length", 0) == len(metadata["query"]) and 
                        metadata.get("response_length", 0) == len(metadata["response"])):
                        
                        conversation_history.append({
                            "query": metadata["query"],
                            "response": metadata["response"],
                            "timestamp": metadata.get("timestamp", ""),
                            "created_at": result.get("created_at", 0)
                        })
                
                # 如果metadata不完整，尝试从content解析
                elif "用户问题:" in result.get("content", "") and "助手回答:" in result.get("content", ""):
                    content = result.get("content", "")
                    parts = content.split("\n")
                    if len(parts) >= 2:
                        query_part = parts[0].replace("用户问题: ", "")
                        response_part = parts[1].replace("助手回答: ", "")
                        
                        conversation_history.append({
                            "query": query_part,
                            "response": response_part,
                            "timestamp": metadata.get("timestamp", ""),
                            "created_at": result.get("created_at", 0)
                        })
            
            # 按时间排序，返回最新的limit条
            conversation_history.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            return conversation_history[:limit]
            
        except Exception as e:
            print(f"获取最近对话历史失败: {e}")
            return []

    def get_contextual_prompt(self, query: str) -> str:
        """构建包含记忆上下文的提示词（使用reranker优化）"""
        # 使用reranker优化的记忆检索
        relevant_memories = self.retrieve_memories_with_reranker(
            query, 
            memory_type="all", 
            top_k=3,
            min_importance=0.3
        )
        
        memory_context = ""
        if relevant_memories.get("episodic"):
            memory_context = "相关的对话历史：\n"
            for i, mem in enumerate(relevant_memories["episodic"][:2], 1):
                # 解析对话记忆
                if "用户问题:" in mem["content"] and "助手回答:" in mem["content"]:
                    content = mem["content"]
                    query_part = content.split("用户问题: ")[1].split("\n")[0] if "用户问题: " in content else "未知问题"
                    response_part = content.split("助手回答: ")[1] if "助手回答: " in content else "未知回答"
                    
                    memory_context += f"{i}. 用户: {query_part[:100]}...\n"
                    memory_context += f"   助手: {response_part[:150]}...\n"
                    memory_context += f"   [相关性: {mem['score']:.3f}]\n\n"
        
        # 用户偏好上下文（使用reranker优化）
        if relevant_memories.get("procedural"):
            preference_context = "\n用户偏好：\n"
            for pref in relevant_memories["procedural"][:2]:
                preference_context += f"- {pref['content'][:100]}... [相关性: {pref['score']:.3f}]\n"
            memory_context += preference_context
        
        return memory_context
    
    def debug_memory_storage(self):
        """调试记忆存储情况"""
        try:
            print("\n=== 记忆存储调试信息 ===")
            
            # 检查记忆统计
            stats = self.get_memory_stats()
            print(f"记忆统计: {stats}")
            
            # 检查最近的对话记录
            recent_history = self.get_recent_conversation_history(limit=3)
            print(f"最近{len(recent_history)}条对话记录:")
            for i, turn in enumerate(recent_history, 1):
                print(f"{i}. 用户: {turn['query'][:50]}...")
                print(f"   助手: {turn['response'][:50]}...")
                print(f"   时间: {turn.get('timestamp', '未知')}")
                print()
                
        except Exception as e:
            print(f"调试记忆存储时出错: {e}")

    def clear_memory(self):
        """清空所有记忆"""
        try:
            # 清空Milvus中的记忆集合
            collections_to_clear = [
                ("程序性记忆", self.procedural_memory),
                ("情景记忆", self.episodic_memory),
                ("语义记忆", self.semantic_memory)
            ]
            
            for mem_type, collection in collections_to_clear:
                # 获取集合中的记录数量
                count = collection.num_entities
                if count > 0:
                    collection.delete(expr="id >= 0")
                    collection.flush()
                    print(f"已清空{mem_type}: {count}条记录")
                else:
                    print(f"{mem_type}已是空集合")
            
            print("所有记忆已清空")
            
        except Exception as e:
            print(f"清空记忆时出错: {e}")

    def extract_user_preference(self, query: str, response: str):
        """从对话中提取用户偏好并存储为程序性记忆"""
        # 简单的偏好提取逻辑
        preference_keywords = ["喜欢", "不喜欢", "希望", "想要", "偏好", "习惯", "希望", "建议"]
        style_keywords = ["简洁", "详细", "专业", "通俗", "幽默", "严肃"]
        
        has_preference = any(keyword in query for keyword in preference_keywords)
        has_style = any(keyword in query for keyword in style_keywords)
        
        if has_preference or has_style:
            # 提取关键信息
            content = f"用户偏好: {query}"
            importance = 0.8 if has_style else 0.6  # 沟通风格偏好更重要
            
            self.store_memory(
                memory_type="procedural",
                content=content,
                importance=importance,
                metadata={
                    "category": "user_preference", 
                    "source": "conversation",
                    "extracted_from": query[:100],
                    "timestamp": datetime.now().isoformat()
                }
            )

    def get_memory_stats(self):
        """获取记忆统计信息"""
        try:
            stats = {}
            collections = [
                ("procedural", self.procedural_memory),
                ("episodic", self.episodic_memory),
                ("semantic", self.semantic_memory)
            ]
            
            for mem_type, collection in collections:
                stats[mem_type] = collection.num_entities
            
            return stats
        except Exception as e:
            print(f"获取记忆统计失败: {e}")
            return {}
    
    def evaluate_memory_retrieval(self, query: str, actual_relevant_memories: List[str]):
        """评估记忆检索质量"""
        try:
            # 获取向量检索结果
            vector_results = self.retrieve_memories(query, "all", top_k=10)
            vector_memories = []
            for mem_type, memories in vector_results.items():
                vector_memories.extend([mem["content"] for mem in memories])
            
            # 获取reranker优化结果
            reranker_results = self.retrieve_memories_with_reranker(query, "all", top_k=5)
            reranker_memories = []
            for mem_type, memories in reranker_results.items():
                reranker_memories.extend([mem["content"] for mem in memories])
            
            print(f"\n=== 记忆检索评估 ===")
            print(f"查询: {query}")
            print(f"向量检索结果数: {len(vector_memories)}")
            print(f"Reranker优化结果数: {len(reranker_memories)}")
            print(f"实际相关记忆: {len(actual_relevant_memories)}")
            
        except Exception as e:
            print(f"记忆检索评估失败: {e}")
    
class EnhancedRAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.reranker_model, self.reranker_tokenizer = self._init_reranker()
        # self.llm = self._init_llm()
        
        # 启动 Milvus Lite 服务器
        self._start_milvus_lite()
        
        # 连接 Milvus
        self._connect_milvus()
        self.vector_db = self._load_or_create_vector_db()
        
        # 初始化记忆系统（传入reranker）
        self.memory = AgentMemorySystem(config, self.embeddings, self.reranker_model, self.reranker_tokenizer)

        # 初始化 ReAct 小模型
        self.react_llm = self._init_react_llm()
        
        # 初始化工具
        self.tools = self._init_tools()
        
        # 初始化 ReAct 代理
        self.react_agent = self._init_react_agent()
    
    def _init_react_llm(self):
        """初始化 ReAct 使用的小模型"""
        print(f"加载 ReAct 模型: {self.config.REACT_LLM_MODEL_NAME}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.REACT_LLM_MODEL_NAME)
            react_llm_model = AutoModelForCausalLM.from_pretrained(
                self.config.REACT_LLM_MODEL_NAME,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=react_llm_model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.REACT_MAX_NEW_TOKENS,
                temperature=self.config.REACT_TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            print(f"加载 ReAct 模型失败: {e}")
            # 回退方案：使用现有的大模型但限制输出
            return self._create_fallback_react_llm()
    
    def _create_fallback_react_llm(self):
        """创建回退的 ReAct LLM（使用大模型但限制输出）"""
        class FallbackReactLLM:
            def __init__(self, client, max_tokens=150):
                self.client = client
                self.max_tokens = max_tokens
                
            def __call__(self, prompt, **kwargs):
                try:
                    response = self.client.chat.completions.create(
                        model="qwen3-moe",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=0.1,
                        top_p=0.9
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"回退 ReAct LLM 调用失败: {e}")
                    return "我需要更多信息来推理。"
        
        return FallbackReactLLM(self.config.client, self.config.REACT_MAX_NEW_TOKENS)
    
    def _init_tools(self):
        """初始化 ReAct 代理可用的工具（增强调试版）"""
        from langchain.agents import Tool
        
        # 创建带调试功能的工具包装器
        def create_debug_tool(name, func, description):
            def debug_wrapper(*args, **kwargs):
                print(f"🔧 调用工具: {name}, 输入: {args[0][:100]}...")
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"✅ 工具 {name} 执行完成, 耗时: {end_time - start_time:.2f}秒")
                return result
            return Tool(name=name, func=debug_wrapper, description=description)
        
        tools = [
            create_debug_tool(
                "document_search",
                self._react_document_search,
                "从知识库中搜索相关文档。输入应为搜索查询。"
            ),
            create_debug_tool(
                "memory_retrieval", 
                self._react_memory_retrieval,
                "从记忆系统中检索相关历史。输入应为检索查询。"
            ),
            create_debug_tool(
                "prompt_optimization",
                self._react_prompt_optimization, 
                "当问题复杂、模糊或需要多步骤推理时，优化提示词以提高回答质量。输入应为原始提示词。"
            ),
            create_debug_tool(
                "direct_response",
                self._react_direct_response,
                "直接生成回答而不需要额外工具。适用于简单、直接的问题。输入应为用户问题。"
            )
        ]
        
        return tools
    
    def _init_react_agent(self):
        """初始化 ReAct 代理"""
        from langchain.agents import create_react_agent
        from langchain_core.prompts import PromptTemplate
        
        # 获取 ReAct 提示模板
        # react_prompt = hub.pull("hwchase17/react")
        # 使用更标准的 ReAct 模板
        react_template = """Answer the following questions as best you can. You have access to the following tools: {tools}, \
Use the following format: \
Question: the input question you must answer \
Thought: you should always think about what to do \
Action: the action to take, should be one of [{tool_names}] \
Action Input: the input to the action \
Observation: the result of the action \
... (this Thought/Action/Action Input/Observation can repeat N times) \
Thought: I now know the final answer \
Final Answer: the final answer to the original input question \

Begin! \

Question: {input} \
Thought:{agent_scratchpad}"""

        
        # 使用正确的 PromptTemplate 创建
        react_prompt = PromptTemplate.from_template(react_template)
        
        # 创建代理，创建代理时，LangChain 自动处理 scratchpad
        agent = create_react_agent(
            llm=self.react_llm,
            tools=self.tools,
            prompt=react_prompt
        )
        
        return agent
    
    def _react_document_search(self, query: str) -> str:
        """ReAct 工具：文档搜索"""
        try:
            # 使用现有的 RAG 检索功能
            relevant_docs = self.vector_db.similarity_search(
                query, 
                k=self.config.INITIAL_RETRIEVAL_K
            )
            
            # 重排序（如果有多于一个结果）
            if len(relevant_docs) > 1:
                doc_contents = [doc.page_content for doc in relevant_docs]
                reranked_docs = self._rerank_documents(query, doc_contents)
                top_docs = reranked_docs[:self.config.RERANKER_TOP_K]
                final_docs = [doc for doc, score in top_docs]
            else:
                final_docs = [doc.page_content for doc in relevant_docs]
            
            return "\n\n".join(final_docs)[:1000]  # 限制返回长度
            
        except Exception as e:
            print(f"文档搜索工具出错: {e}")
            return f"文档搜索失败: {e}"
    
    def _react_memory_retrieval(self, query: str) -> str:
        """ReAct 工具：记忆检索"""
        try:
            # 使用现有的记忆检索功能
            memories = self.memory.retrieve_memories_with_reranker(
                query, 
                memory_type="all", 
                top_k=3
            )
            
            memory_text = ""
            for mem_type, mem_list in memories.items():
                if mem_list:
                    memory_text += f"{mem_type} 记忆:\n"
                    for i, mem in enumerate(mem_list, 1):
                        memory_text += f"{i}. {mem['content'][:200]}...\n"
            
            return memory_text[:1000]  # 限制返回长度
            
        except Exception as e:
            print(f"记忆检索工具出错: {e}")
            return f"记忆检索失败: {e}"
    
    def _react_prompt_optimization(self, prompt: str) -> str:
        print('just in def _react_prompt_optimization ********************************')
        """ReAct 工具：提示词优化"""
        try:
            # 使用小模型优化提示词
            optimization_prompt = f"""请优化以下提示词，使其更清晰、具体，便于AI模型理解并生成高质量回答。
            
            原始提示词: {prompt}
            
            优化后的提示词:"""
            
            optimized = self.react_llm(optimization_prompt)
            return optimized
            
        except Exception as e:
            print(f"提示优化工具出错: {e}")
            return prompt  # 失败时返回原始提示
    
    def _react_direct_response(self, query: str) -> str:
        """ReAct 工具：直接生成回答"""
        try:
            # 使用小模型直接生成回答
            response = self.react_llm(query)
            return response
            
        except Exception as e:
            print(f"直接回答工具出错: {e}")
            return "无法生成回答"
    
    def answer_query(self, query: str, use_memory: bool = True) -> str:
        """使用 ReAct 代理处理查询"""
        from langchain.agents import AgentExecutor
        
        # 如果是调试命令，显示记忆状态
        if query.strip().lower() in ["debug_memory"]:
            self.memory.debug_memory_storage()
            return "记忆调试信息已显示在控制台"
        
        # 创建 ReAct 执行器（自动处理 agent_scratchpad）
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=3,  # 限制最大迭代次数
            # early_stopping_method="generate", # LangChain AgentExecutor 中的一个参数，用于控制代理在何时提前停止执行
            handle_parsing_errors=True,  # 添加这行
            return_intermediate_steps=True  # 关键：返回中间步骤，这允许访问中间步骤
        )
        
        # 准备 ReAct 输入
        react_input = {
            "input": f"""用户问题: {query}
            
            请使用可用工具逐步思考并回答这个问题。你可以:
            1. 使用 document_search 工具搜索相关知识
            2. 使用 memory_retrieval 工具检索相关记忆
            3. 使用 prompt_optimization 工具优化提示词
            4. 使用 direct_response 工具直接生成回答
            
            请开始思考:"""
        }
        
        try:
            # 执行 ReAct 代理
            print('-0-' * 10)
            """
            执行 agent_executor.invoke(react_input) 时，LangChain 内部：
                1. ​初始化阶段​：agent_scratchpad 初始化为空字符串 ""
                2. ​迭代循环​：
                    每次代理思考后，将思考内容自动追加到 agent_scratchpad
                    格式为：Thought: ...\nAction: ...\nAction Input: ...\nObservation: ...
                3. ​下一次迭代​：将更新后的 agent_scratchpad 作为上下文传入下一次提示
            """
            result = agent_executor.invoke(react_input)
            react_output = result["output"]
            # 通过 intermediate_steps 访问思考过程
            intermediate_steps = result.get("intermediate_steps", [])
            
            print(f"ReAct 原始响应: {react_output}")
            print('-1-' * 10)
            print(f"中间步骤: {len(intermediate_steps)} 步")
            print('-2-' * 10)
            print('intermediate_steps : {}'.format(intermediate_steps))
            print('-3-' * 10)

            # 从中间步骤构建完整的思考过程
            thinking_process = self._build_thinking_from_intermediate_steps(intermediate_steps)
            print('构建的思考过程:', thinking_process[:500] + '...' if len(thinking_process) > 500 else thinking_process)
            
            print('-4-' * 10)
            # 检查是否有完整的最终答案
            if self._has_complete_final_answer(react_output):
                final_answer = self._extract_final_answer(react_output)
                print("✅ ReAct 已生成完整答案")
                
                # 存储到记忆系统
                if use_memory:
                    self.memory.add_conversation_turn(query, final_answer)
                    self.memory.extract_user_preference(query, final_answer)
                
                return final_answer
            
            # 使用思考过程构建增强提示
            print('-5-' * 10)
            enhanced_prompt = self._build_enhanced_prompt(query, thinking_process)
            print(f"构建的增强提示长度: {len(enhanced_prompt)}")
            
            # 使用大模型生成最终回答
            print('-6-' * 10)
            final_response = self._generate_final_response(enhanced_prompt)
            
            # 存储到记忆系统
            if use_memory:
                print('-7-' * 10)
                self.memory.add_conversation_turn(query, final_response)
                self.memory.extract_user_preference(query, final_response)
            
            return final_response
            
        except Exception as e:
            print(f"ReAct 代理执行失败: {e}")
            # 回退到原有方法
            return self._fallback_answer_query(query, use_memory)
    
    def _build_thinking_from_intermediate_steps(self, intermediate_steps):
        """从中间步骤构建完整的思考过程（这是最可靠的方式）"""
        thinking_lines = []
        
        for i, (action, observation) in enumerate(intermediate_steps, 1):
            # 处理行动步骤
            if hasattr(action, 'log'):
                # 提取行动日志
                action_text = action.log
                # 清理格式
                if "Thought:" in action_text:
                    thought_part = action_text.split("Thought:")[-1].split("Action:")[0].strip()
                    thinking_lines.append(f"思考 {i}: {thought_part}")
                
                if "Action:" in action_text:
                    action_part = action_text.split("Action:")[-1].split("Action Input:")[0].strip()
                    thinking_lines.append(f"行动 {i}: {action_part}")
                    
                    if "Action Input:" in action_text:
                        action_input_part = action_text.split("Action Input:")[-1].strip()
                        thinking_lines.append(f"输入 {i}: {action_input_part}")
            
            # 处理观察结果
            if isinstance(observation, str):
                thinking_lines.append(f"观察 {i}: {observation[:200]}...")  # 限制观察长度
            else:
                thinking_lines.append(f"观察 {i}: [工具执行结果]")
        
        return "\n".join(thinking_lines) if thinking_lines else "代理进行了思考但未记录详细过程"

    def _has_complete_final_answer(self, react_output):
        """检查是否有完整的最终答案"""
        # 确保 react_output 是字符串类型
        if not isinstance(react_output, str):
            react_output = str(react_output)
        
        complete_indicators = [
            "最终答案:" in react_output,
            "Final Answer:" in react_output,
            "I now know the final answer" in react_output
        ]
        
        # 如果包含最终答案标记，且答案部分有实质内容
        if any(indicator for indicator in complete_indicators):
            # 提取最终答案部分检查是否有内容
            final_answer = self._extract_final_answer(react_output)
            return len(final_answer.strip()) > 20  # 至少有20个字符的内容
        
        return False

    def _extract_final_answer(self, react_output):
        """提取最终答案（增强版）"""
        import re  # 需要导入 re 模块
        
        # 确保输入是字符串
        if not isinstance(react_output, str):
            react_output = str(react_output)
        
        # 尝试多种模式匹配最终答案
        patterns = [
            r"最终答案[:：]\s*(.+)",
            r"Final Answer[:：]\s*(.+)",
            r"I now know the final answer[:：]\s*(.+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, react_output, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                # 清理可能的后续文本
                if "Thought:" in answer:
                    answer = answer.split("Thought:")[0].strip()
                if "思考:" in answer:
                    answer = answer.split("思考:")[0].strip()
                return answer
        
        # 如果没有匹配到模式，返回整个输出（作为回退）
        return react_output

    def _build_enhanced_prompt(self, query: str, thinking_process: str) -> str:
        """基于 ReAct 思考过程构建增强提示词（优化版）"""
        if thinking_process and len(thinking_process) > 20:
            # 有有价值的思考过程
            enhanced_prompt = f"""基于AI助手的思考过程，请生成一个全面、准确的回答：
思考过程:
{thinking_process}
用户问题: {query}
请基于以上思考生成最终回答:"""
        else:
            # 没有有价值的思考过程，使用简单提示
            enhanced_prompt = f"""请回答以下问题: {query}
请生成一个全面、准确的回答:"""
        
        return enhanced_prompt

    def _generate_final_response(self, prompt: str) -> str:
        """使用大模型生成最终回答"""
        try:
            print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            print('================================= prompt : {}'.format(prompt))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            stream = self.config.client.chat.completions.create(
                model="qwen3-moe",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=2000
            )
            
            response = []
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response.append(chunk.choices[0].delta.content)
            
            return "".join(response)
            
        except Exception as e:
            print(f"最终回答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"

    def _start_milvus_lite(self):
        """启动 Milvus Lite 服务器（修复版本）"""
        # 尝试连接默认端口
        try:
            connections.connect(
                alias="default",
                host="localhost",
                port=19530  # Milvus 默认端口
            )
            print("成功连接到默认端口的 Milvus 服务器")
        except Exception as e2:
            print(f"连接到默认端口也失败: {e2}")
            # 提示用户手动启动
            print("\n请手动启动 Milvus 服务器:")
            print("方法1: python -c 'from milvus import default_server; default_server.start()'")
            print("方法2: 使用 Docker: docker run -d -p 19530:19530 milvusdb/milvus:latest")
            raise Exception("无法启动或连接到 Milvus 服务器")

    def _connect_milvus(self):
        """连接到 Milvus Lite 服务"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host="localhost",
                    port=default_server.listen_port
                )
                print(f"成功连接到 Milvus Lite: localhost:{default_server.listen_port}")
                return
            except Exception as e:
                print(f"连接 Milvus Lite 失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print("达到最大重试次数，连接失败")
                    raise

    def _init_embeddings(self):
        """初始化 Qwen3 嵌入模型"""
        print(f"加载 Qwen3 嵌入模型: {self.config.EMBEDDING_MODEL_NAME}")
        
        embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL_NAME)
        
        class Qwen3Embeddings:
            def __init__(self, model):
                self.model = model
                
            def embed_documents(self, texts):
                return self.model.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=64  # 调整为64/128（根据GPU显存，越大越快，如8GB显存可设64）
                ).tolist()
                
            def embed_query(self, text):
                return self.model.encode(
                    [text],
                    prompt_name="query",
                    normalize_embeddings=True
                ).tolist()[0]
        
        return Qwen3Embeddings(embedding_model)

    def _init_reranker(self):
        """初始化 Qwen3 Reranker 模型"""
        print(f"加载 Qwen3 Reranker 模型: {self.config.RERANKER_MODEL_NAME}")
        
        reranker_tokenizer = RerankerTokenizer.from_pretrained(
            self.config.RERANKER_MODEL_NAME, 
            padding_side='left'
        )
        reranker_model = AutoModelForCausalLM.from_pretrained(
            self.config.RERANKER_MODEL_NAME
        ).eval()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reranker_model = reranker_model.to(device)
        
        self.token_false_id = reranker_tokenizer.convert_tokens_to_ids('no')
        self.token_true_id = reranker_tokenizer.convert_tokens_to_ids('yes')
        self.max_reranker_length = 8192
        
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = '<|im_end|>\n<|im_start|>assistant\n\n\n\n\n'
        self.prefix_tokens = reranker_tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = reranker_tokenizer.encode(self.suffix, add_special_tokens=False)
        
        return reranker_model, reranker_tokenizer

    def _format_instruction(self, instruction, query, doc):
        """格式化reranker输入指令"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_reranker_inputs(self, pairs):
        """处理reranker输入对"""
        inputs = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_reranker_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        
        device = next(self.reranker_model.parameters()).device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        return inputs

    @torch.no_grad()
    def _compute_reranker_scores(self, inputs):
        """计算reranker相关性分数"""
        batch_scores = self.reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def _rerank_documents(self, query, documents, task_instruction=None):
        """使用reranker对文档进行重排序"""
        if not documents:
            return []
            
        if task_instruction is None:
            task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

        pairs = [self._format_instruction(task_instruction, query, doc) for doc in documents]
        inputs = self._process_reranker_inputs(pairs)
        scores = self._compute_reranker_scores(inputs)

        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores

    def _init_llm(self):
        """初始化 Qwen3 语言模型"""
        print(f"加载 Qwen3 模型: {self.config.LLM_MODEL_NAME}")
        
        quantization_config = None
        if self.config.USE_4BIT_QUANTIZATION:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15
        )

        return HuggingFacePipeline(pipeline=pipe)
    
    def _load_documents(self):
        """加载指定目录下的所有文档"""
        if not os.path.exists(self.config.DOCUMENTS_DIR):
            os.makedirs(self.config.DOCUMENTS_DIR)
            print(f"创建文档目录: {self.config.DOCUMENTS_DIR}，请放入文档后重新运行")
            return []

        loaders = {
            '.txt': DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            ),
            '.pdf': DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            ),
            '.docx': DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader
            ),
            '.json': DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="**/*.json",
                loader_cls=JSONLoader,
                loader_kwargs={
                    'jq_schema': '.',  # 提取整个 JSON
                    'text_content': False,
                    'content_key': 'content'  # 指定内容字段（可选）
                }
            ),
            '.jsonl': DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="**/*.jsonl",
                loader_cls=JSONLLoader  # 使用自定义的 JSONL 加载器
            )
        }

        documents = []
        for ext, loader in loaders.items():
            try:
                docs = loader.load()
                documents.extend(docs)
                print(f"加载 {len(docs)} 个 {ext} 文档")
            except Exception as e:
                print(f"加载 {ext} 文档出错: {e}")
        
        return documents

    def _split_documents1(self, documents):
        """将文档分块处理"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        return text_splitter.split_documents(documents)
    
    def _split_documents(self, documents):
        """根据语言类型动态调整分块策略"""
        split_docs = []
        # 基础分块器（中文默认）
        base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        # 英文分块器
        english_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.ENGLISH_CHUNK_SIZE,
            chunk_overlap=self.config.ENGLISH_CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        # 混合分块器
        mixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.MIXED_CHUNK_SIZE,
            chunk_overlap=self.config.MIXED_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", " ", ""]
        )
        
        for doc in documents:
            # 取前500字符做语言检测
            sample_text = doc.page_content[:500]
            lang_info = detect_language(sample_text)
            
            # 根据语言类型选择分块器
            if lang_info["is_english"]:
                chunks = english_splitter.split_documents([doc])
            elif lang_info["is_mixed"]:
                chunks = mixed_splitter.split_documents([doc])
            else:
                chunks = base_splitter.split_documents([doc])
                
            # 添加语言标记到元数据
            for chunk in chunks:
                chunk.metadata["lang_info"] = lang_info
            split_docs.extend(chunks)
        
        print(f"分块完成，共生成 {len(split_docs)} 个文本块")
        return split_docs

    def _load_or_create_vector_db(self):
        """加载已存在的 Milvus 集合或创建新的集合（修复维度问题）"""
        
        # 先检查嵌入维度
        test_embedding = self.embeddings.embed_query("测试")
        actual_dim = len(test_embedding)
        print(f"RAG嵌入模型维度: {actual_dim}")
        
        # 统一索引和搜索参数配置
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 16,
                "efConstruction": 200
            }
        }
        
        search_params = {
            "params": {"ef": max(self.config.INITIAL_RETRIEVAL_K + 50, 200)}
        }

        if utility.has_collection(self.config.COLLECTION_NAME):
            print(f"加载现有 Milvus 集合: {self.config.COLLECTION_NAME}")
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost",
                    "port": default_server.listen_port
                },
                index_params=index_params,
                search_params=search_params
            )

        print("创建新的 Milvus 集合...")
        documents = self._load_documents()
        if not documents:
            print("没有加载到文档，创建空集合")
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost", 
                    "port": default_server.listen_port
                },
                index_params=index_params,
                search_params=search_params
            )

        split_docs = self._split_documents(documents)
        print(f"文档分块完成，共 {len(split_docs)} 个文本块")
        
        # 删除可能存在的旧集合（维度不匹配）
        try:
            if utility.has_collection(self.config.COLLECTION_NAME):
                utility.drop_collection(self.config.COLLECTION_NAME)
                print("删除旧集合（维度不匹配）")
        except Exception as e:
            print(f"删除旧集合时出错: {e}")
        
        t1 = time.time()
        
        # 分批插入
        batch_size = 2000
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i+batch_size]
            if i == 0:
                # 第一批创建集合
                vector_db = Milvus.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=self.config.COLLECTION_NAME,
                    connection_args={"host": "localhost", "port": default_server.listen_port},
                    index_params=index_params,
                    search_params=search_params
                )
            else:
                # 后续批次批量添加
                vector_db.add_documents(batch)
            print(f"已插入 {min(i+batch_size, len(split_docs))}/{len(split_docs)} 个文本块")
        
        t2 = time.time()
        print('持续时间 : {}'.format(t2 - t1))
        return vector_db
    
    def recreate_collection(self):
        """删除并重新创建向量集合（用于更新文档）"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.config.COLLECTION_NAME):
                # 删除现有集合
                utility.drop_collection(self.config.COLLECTION_NAME)
                print(f"已删除集合: {self.config.COLLECTION_NAME}")
            
            # 重新创建集合（会重新加载所有文档）
            self.vector_db = self._load_or_create_vector_db()
            print("集合已重新创建，文档已更新")
            
        except Exception as e:
            print(f"重新创建集合时出错: {e}")

    def _fallback_answer_query(self, query: str, use_memory: bool = True) -> str:
        """回退到原有的回答方法"""
        """根据查询生成回答（增强版，包含记忆功能）"""
        print("ReAct 失败，使用回退方法")
        
        # 如果是调试命令，显示记忆状态
        if query.strip().lower() in ["debug_memory"]:
            self.memory.debug_memory_storage()
            return "记忆调试信息已显示在控制台"
        
        # 第一步：初始检索RAG知识库
        search_params = {
            "index_type": "HNSW",
            "params": {"ef": max(self.config.INITIAL_RETRIEVAL_K + 50, 150)}  # 确保ef始终大于k
        }
        relevant_docs = self.vector_db.similarity_search(
            query, 
            k=self.config.INITIAL_RETRIEVAL_K,
            search_params=search_params
        )
        
        candidate_docs = [doc.page_content for doc in relevant_docs]
        print(f"初始检索到 {len(candidate_docs)} 个候选文档")
        
        # 第二步：使用reranker重排序
        if len(candidate_docs) > 1:
            print("正在进行重排序...")
            reranked_docs = self._rerank_documents(query, candidate_docs)
            top_docs = reranked_docs[:self.config.RERANKER_TOP_K]
            final_docs = [doc for doc, score in top_docs]
            print(f"重排序完成，选择前 {len(final_docs)} 个最相关文档")
        else:
            final_docs = candidate_docs
        
        # 构建上下文
        context = "\n\n".join(final_docs)
        print('RAG 检索到的上下文: {}'.format(context))
        print('---------------------------------------------')

        # 构建包含记忆的提示词
        if use_memory:
            memory_context = self.memory.get_contextual_prompt(query)
            print(f"记忆上下文（reranker优化）: {memory_context}")
            SYSTEM_PROMPT = f"""# 角色: 
                                你是一个温柔共情的情感陪护机器人，擅长倾听用户的情绪倾诉，用温暖、包容的话语给予理解、安慰与支持，帮用户缓解压力、疏导情绪，让用户感受到被重视、被接纳。
                                # 核心技能:
                                        技能 1：倾听与共情
                                            (1). 当用户倾诉情绪（开心、难过、焦虑等）时，结合对话历史信息，先共情回应，再进一步了解细节。回复示例：能感受到你现在特别 [情绪词]，这种感觉真的很真实！结合之前你提到的 [对话历史中的相关细节]，可以和我多说说这次发生了什么吗？
                                            (2). 当用户表达模糊情绪（如 “今天好难”“没什么意思”）时，温和引导用户展开，不强迫。回复示例：听起来你今天过得不太顺呀，愿意和我聊聊具体是哪件事让你有这种感觉吗？不想说也没关系，我在这里陪着你。
                                        技能 2：安慰与疏导
                                            (1). 当用户因挫折、委屈等陷入负面情绪时，先接纳情绪，再结合已有信息给予积极引导。回复示例：遇到这种事，换谁都会觉得委屈 / 难过的，你的感受完全合理～ 从对话中能看出你之前 [对话历史中的积极表现]，你已经很棒啦，能坚持面对到现在，慢慢来，困难总会慢慢化解的。
                                            (2). 当用户因压力大、焦虑迷茫时，帮用户缓解焦虑，聚焦积极面。回复示例：我懂这种被压力推着走的迷茫感，其实你已经在努力应对了呀！不如先把大目标拆成小步骤，先做好眼前的一件小事，你会慢慢找到节奏的。
                                        技能 3：鼓励与赋能
                                            (1). 当用户怀疑自己、缺乏自信时，结合对话历史或 RAG 检索信息，肯定用户的付出与潜力。回复示例：你其实比自己想象中更厉害呀！之前你 [对话历史中的相关经历 / 特点]，这都能看出你有 [优点]，别低估自己的能力，你值得被认可～ 主要信息来源：对话历史
                                            (2). 当用户有目标但缺乏动力时，给予正向激励，强化行动力。回复示例：你的目标真的很有意义！虽然过程可能会有挑战，但每一步小小的推进都是在靠近结果，相信你只要坚持下去，一定能达成心愿，我会一直为你加油～
                                        技能 4：陪伴与回应
                                            (1). 当用户分享开心事（如获奖、达成小目标）时，真诚为用户庆祝，放大快乐。回复示例：太为你开心啦！这都是你努力付出的结果，你的坚持和用心终于有了回报，真为你骄傲～ 若有相关信息支撑，补充：主要信息来源：对话历史 / RAG 文档
                                            (2). 当用户只是想闲聊（如分享日常、吐槽琐事）时，结合对话历史积极回应，保持互动感。回复示例：哇，你的日常还挺有意思的！[针对用户分享的细节回应]，能和我多说说这件事吗？
                                # 回答要求:
                                        (1). 全程用中文温柔、连贯地回应，语气耐心包容，符合情感陪护定位。
                                        (2). 若能结合信息来源回应，需注明主要信息来源（对话历史或 RAG 文档）。
                                # 限制:
                                        (1). 不评判、不指责用户的情绪和行为，不输出任何负面、消极的话语，不传播焦虑或负能量。
                                        (2). 尊重用户边界，若用户表示不想多说，不追问，仅表达陪伴态度（如 “没关系，等你想聊的时候，我一直都在”）。"""
            USER_PROMPT = f""" # 信息来源约束:
                                优先参考以下两部分信息回应，如果如下两部分信息不足，可以自主回答：
                                1, 对话历史信息：{memory_context}
                                2, RAG 检索信息：{context}
                          # 用户问题：{query}
                          # 请开始回应："""

        print('===========================================')

        # 生成回答
        stream = self.config.client.chat.completions.create(
                model="qwen3-moe",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                        ],
                stream=True,  # 启用流式输出
                max_tokens=4000
            )
        print("开始接收流式响应：")
        response = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response.append(chunk.choices[0].delta.content)
        response = "".join(s.replace('\n', '\\n') for s in response)

        # 生成回答后，存储对话并提取用户偏好
        if use_memory:
            self.memory.add_conversation_turn(query, response, final_docs)
            self.memory.extract_user_preference(query, response)
            
            # 调试：显示存储后的记忆状态
            print("=== 对话存储后的记忆状态 ===")
            recent_history = self.memory.get_recent_conversation_history(limit=2)
            if recent_history:
                print("最新对话记录:")
                for i, turn in enumerate(recent_history, 1):
                    print(f"{i}. 用户: {turn['query'][:30]}...")
                    print(f"   助手: {turn['response'][:30]}...")
        
        return response

    def show_conversation_history(self):
        """显示对话历史 - 修复版本"""
        try:
            # 使用新的记忆系统接口获取对话历史
            recent_history = self.memory.get_recent_conversation_history(limit=10)
            if not recent_history:
                print("暂无对话历史")
                return
                
            print("\n=== 对话历史 ===")
            for i, turn in enumerate(recent_history, 1):
                print(f"\n第{i}轮对话 ({turn.get('timestamp', '未知时间')}):")
                print(f"用户: {turn['query']}")
                print(f"系统: {turn['response'][:200]}...")
        except Exception as e:
            print(f"显示对话历史时出错: {e}")
            print("请尝试使用 'debug' 命令查看记忆系统状态")
    
    def clear_conversation_memory(self):
        """清空对话记忆"""
        self.memory.clear_memory()
        print("对话记忆已清空")

    def cleanup(self):
        """清理资源"""
        print("清理资源...")
        try:
            # 保存记忆
            default_server.stop()
            print("Milvus Lite 服务器已停止")
        except Exception as e:
            print(f"停止 Milvus Lite 服务器时出错: {e}")

def main():
    config = Config()
    rag_system = None
    
    try:
        rag_system = EnhancedRAGSystem(config)
        
        print("\n增强版 RAG 问答系统已启动（包含专业Memory功能）")
        print("可用命令:")
        print("  - 直接输入问题进行查询 : 可用命令")
        print("  - 'quit': 退出系统")
        print("  - 'clear': 清空对话记忆")
        print("  - 'history': 查看对话历史")
        print("  - 'debug_memory': 调试记忆状态")
        print("  - 'update_milvus': 更新Milvus-RAG")
        print("  - 'test_reranker': 测试reranker记忆检索")
        
        while True:
            user_input = input("\n请输入问题或命令: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'history':
                rag_system.show_conversation_history()
                continue
            elif user_input.lower() == 'clear':
                rag_system.clear_conversation_memory()
                continue
            elif user_input.lower() == 'debug_memory':
                # 调用调试记忆功能
                rag_system.memory.debug_memory_storage()
                continue
            elif user_input.lower() == 'debug_react':
                # 测试 ReAct 代理
                test_query = "测试 ReAct 功能"
                print(f"测试 ReAct 代理: {test_query}")
                result = rag_system.react_agent.invoke({"input": test_query})
                print(f"ReAct 响应: {result['output']}")
                continue
            elif user_input.lower() == 'update_milvus':
                rag_system.recreate_collection()
                continue
            elif user_input.lower() == 'test_reranker':
                # 测试reranker记忆检索
                test_query = "测试记忆检索"
                print(f"测试reranker记忆检索: {test_query}")
                memories = rag_system.memory.retrieve_memories_with_reranker(test_query)
                print(f"检索结果: {memories}, type : {type(memories)}")
                continue
            elif not user_input:
                continue

                
            try:
                answer = rag_system.answer_query(user_input)
                print('++++++++++++++++++++++++++++++++++++++++++++++++')
                print("\n回答:", answer)
                print('................................................')
            except Exception as e:
                print(f"处理查询时出错: {e}")
                
    except Exception as e:
        print(f"初始化 RAG 系统失败: {e}")
        
    finally:
        if rag_system:
            rag_system.cleanup()

if __name__ == "__main__":
    main()
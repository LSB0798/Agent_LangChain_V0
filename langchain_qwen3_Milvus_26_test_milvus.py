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

    # 新增：并发性能测试的默认配置
    DEFAULT_CONCURRENT_TEST_QUERY = "深度学习"  # 默认测试查询文本
    DEFAULT_CONCURRENCY_LEVEL = 100            # 默认并发数
    MAX_CONCURRENCY_LEVEL = 1000               # 允许的最大并发数（安全上限）

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
    
    def answer_query(self, query: str, use_memory: bool = True) -> str:
        """简化版：直接使用RAG检索和记忆系统回答问题"""
        # 如果是调试命令，显示记忆状态
        if query.strip().lower() in ["debug_memory"]:
            self.memory.debug_memory_storage()
            return "记忆调试信息已显示在控制台"
        
        # 调用原有的回退方法，它已经包含了完整的RAG检索逻辑
        return self._fallback_answer_query(query, use_memory)
    
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
        # print('RAG 检索到的上下文: {}'.format(context))
        # 构建上下文，并改进日志输出
        if len(final_docs) > 0:
            print("\n📄 RAG 检索到的上下文（共 {} 个相关文档片段）:".format(len(final_docs)))
            print("-" * 60)
            
            context_parts = []
            for i, doc_content in enumerate(final_docs, 1):
                # 尝试从文档中提取元数据信息（如果有的话）
                doc_info = f"文档片段 [{i}]"
                
                # 如果reranked_docs存在且包含分数信息，可以显示相关性分数
                if 'reranked_docs' in locals() and i <= len(reranked_docs):
                    score = reranked_docs[i-1][1]  # 获取相关性分数
                    doc_info += f" [相关性分数: {score:.3f}]"
                
                # 截取文档内容的前200个字符作为预览
                preview = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content
                
                print(f"{doc_info}:")
                print(f"  {preview}")
                print()
                
                context_parts.append(doc_content)
            
            context = "\n\n".join(context_parts)
            print("-" * 60)
            print(f"✅ 以上 {len(final_docs)} 个文档片段将用于生成回答")
        else:
            print("⚠️  未检索到相关文档")
            context = ""
        print('---------------------------------------------')

        return context

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
    
    def _execute_single_search(self, task_id: int, query: str, use_memory: bool) -> Dict[str, Any]:
        """
        执行单次搜索任务（用于并发测试的工作单元）。
        返回包含状态、耗时、结果长度等信息的字典。
        """
        result = {
            "task_id": task_id,
            "status": "unknown",
            "elapsed_time": 0.0,
            "answer_length": 0,
            "error": None
        }
        start_time = time.time()
        
        try:
            # 执行核心的查询流程，禁用记忆以确保测试环境纯净、可复现
            answer = self.answer_query(query, use_memory=use_memory)
            end_time = time.time()
            
            result["status"] = "success"
            result["elapsed_time"] = end_time - start_time
            result["answer_length"] = len(answer)
            # 可选：记录答案前100字符用于调试，避免日志过长
            # result["answer_preview"] = answer[:100] + "..." if len(answer) > 100 else answer
            
        except Exception as e:
            end_time = time.time()
            result["status"] = "failed"
            result["elapsed_time"] = end_time - start_time
            result["error"] = str(e)
            print(f"  ❌ 任务 {task_id} 执行失败: {e}")
        
        return result

    def run_concurrent_performance_test(self, query: str, concurrency: int = 100) -> Dict[str, Any]:
        """
        执行并发性能测试。
        
        Args:
            query: 要搜索的文本
            concurrency: 并发数（线程数）
        
        Returns:
            包含整体测试结果的字典
        """
        print(f"\n🚀 开始并发性能测试")
        print(f"   测试查询: '{query}'")
        print(f"   并发数: {concurrency}")
        print("-" * 60)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_results = []
        overall_start_time = time.time()
        
        # 使用 ThreadPoolExecutor 管理并发线程
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务到线程池
            future_to_task = {}
            for i in range(concurrency):
                # 为每个任务生成一个唯一ID，查询文本可稍作修改以增加差异性（可选）
                task_query = f"{query} [任务{i+1}]"
                future = executor.submit(self._execute_single_search, i+1, task_query, use_memory=False)
                future_to_task[future] = i+1
            
            # 等待并收集所有任务的结果
            completed = 0
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    task_result = future.result(timeout=60)  # 设置单任务超时60秒
                    all_results.append(task_result)
                except Exception as e:
                    # 处理任务执行异常（如超时）
                    all_results.append({
                        "task_id": task_id,
                        "status": "timeout_or_error",
                        "elapsed_time": 0.0,
                        "answer_length": 0,
                        "error": f"任务执行或获取结果异常: {e}"
                    })
                completed += 1
                # 每完成10%的任务打印一次进度
                if completed % max(1, concurrency // 10) == 0 or completed == concurrency:
                    print(f"  进度: {completed}/{concurrency} ({completed/concurrency*100:.0f}%)")
        
        overall_end_time = time.time()
        total_test_time = overall_end_time - overall_start_time
        
        # 分析测试结果
        successful = [r for r in all_results if r["status"] == "success"]
        failed = [r for r in all_results if r["status"] != "success"]
        
        # 计算性能指标
        if successful:
            elapsed_times = [r["elapsed_time"] for r in successful]
            avg_time = sum(elapsed_times) / len(elapsed_times)
            min_time = min(elapsed_times)
            max_time = max(elapsed_times)
            # 计算吞吐量 (Queries Per Second)
            qps = len(successful) / total_test_time
        else:
            avg_time = min_time = max_time = qps = 0
        
        # 打印详细测试报告
        print("\n" + "="*60)
        print("📊 并发性能测试报告")
        print("="*60)
        print(f"总测试时间: {total_test_time:.2f} 秒")
        print(f"并发任务总数: {concurrency}")
        print(f"成功任务数: {len(successful)}")
        print(f"失败/超时任务数: {len(failed)}")
        print(f"成功率: {len(successful)/concurrency*100:.1f}%")
        print(f"系统吞吐量 (QPS): {qps:.2f}")
        print(f"平均响应时间: {avg_time:.3f} 秒")
        print(f"最快响应: {min_time:.3f} 秒")
        print(f"最慢响应: {max_time:.3f} 秒")
        
        if successful:
            answer_lengths = [r["answer_length"] for r in successful]
            print(f"平均答案长度: {sum(answer_lengths)/len(answer_lengths):.0f} 字符")
            print(f"最短答案: {min(answer_lengths)} 字符")
            print(f"最长答案: {max(answer_lengths)} 字符")
        
        # 打印响应时间分布
        print(f"\n📈 响应时间分布:")
        time_bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, float('inf'))]
        for low, high in time_bins:
            count = sum(1 for r in successful if low <= r['elapsed_time'] < high)
            if count > 0:
                high_str = f"{high:.1f}" if high != float('inf') else "∞"
                print(f"  {low:.1f}-{high_str}秒: {count} 个任务 ({count/len(successful)*100:.1f}%)")
        
        if failed:
            print(f"\n⚠️  失败任务详情 (前5个):")
            for fail in failed[:5]:
                print(f"  任务 {fail.get('task_id')}: {fail.get('error', '未知错误')}")
        
        print("="*60)
        
        # 返回完整的测试结果，便于进一步分析或记录
        return {
            "query": query,
            "concurrency": concurrency,
            "total_test_time": total_test_time,
            "success_count": len(successful),
            "failure_count": len(failed),
            "success_rate": len(successful)/concurrency*100 if concurrency > 0 else 0,
            "qps": qps,
            "avg_response_time": avg_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "all_results": all_results
        }

def main():
    config = Config()
    rag_system = None
    
    try:
        rag_system = EnhancedRAGSystem(config)
        
        # 修改系统启动提示，明确说明只接收指令
        print("\n简化版 RAG 系统指令控制台已启动")
        print("======================================")
        print("可用指令:")
        print("  - 'test_performance': 运行性能测试（使用固定查询）")
        print("  - 'history': 查看对话历史")
        print("  - 'clear': 清空对话记忆")
        print("  - 'debug_memory': 调试记忆状态")
        print("  - 'update_milvus': 重新创建Milvus集合（更新文档）")
        print("  - 'quit': 退出系统")
        print("======================================")
        print("提示：本系统已设置为指令模式，将不再直接回答问题。")
        
        while True:
            user_input = input("\n请输入指令: ").strip().lower()
            
            if user_input == 'quit':
                print("正在退出系统...")
                break
            elif user_input == 'history':
                rag_system.show_conversation_history()
                continue
            elif user_input == 'clear':
                rag_system.clear_conversation_memory()
                continue
            elif user_input == 'debug_memory':
                rag_system.memory.debug_memory_storage()
                continue
            elif user_input == 'update_milvus':
                rag_system.recreate_collection()
                continue
            elif user_input.startswith('concurrent_test:'):
                # 解析指令格式: concurrent_test[:查询文本[:并发数]]
                # 示例1: concurrent_test                  (使用所有默认值)
                # 示例2: concurrent_test:机器学习         (使用默认并发数)
                # 示例3: concurrent_test:机器学习:50      (指定查询和并发数)
                # 示例4: concurrent_test::20              (使用默认查询，指定并发数)
                parts = user_input.split(':', 2)  # 最多分割成3部分
                
                # 1. 确定测试查询文本
                if len(parts) > 1 and parts[1].strip():
                    test_query = parts[1].strip()
                else:
                    test_query = config.DEFAULT_CONCURRENT_TEST_QUERY
                    print(f"⚠️  使用默认测试查询: '{test_query}'")
                
                # 2. 确定并发数
                concurrency = config.DEFAULT_CONCURRENCY_LEVEL  # 默认值
                if len(parts) > 2 and parts[2].strip():
                    try:
                        user_concurrency = int(parts[2].strip())
                        if 1 <= user_concurrency <= config.MAX_CONCURRENCY_LEVEL:
                            concurrency = user_concurrency
                        else:
                            print(f"⚠️  并发数 {user_concurrency} 超出允许范围(1-{config.MAX_CONCURRENCY_LEVEL})，使用默认值: {concurrency}")
                    except ValueError:
                        print(f"⚠️  并发数格式错误，使用默认值: {concurrency}")
                else:
                    print(f"⚠️  使用默认并发数: {concurrency}")
                
                # 执行并发性能测试
                print(f"\n🚀 开始并发性能测试")
                print(f"   测试查询: '{test_query}'")
                print(f"   并发数: {concurrency}")
                print("-" * 60)
                
                try:
                    rag_system.run_concurrent_performance_test(test_query, concurrency)
                except Exception as e:
                    print(f"❌ 并发性能测试执行失败: {e}")
                continue
            else:
                # 如果输入的不是有效指令，提示用户
                print(f"未知指令: '{user_input}'")
                print("请输入 'quit', 'history', 'clear', 'debug_memory', 'update_milvus' 或 'test_performance' 其中之一。")
                continue
                
    except Exception as e:
        print(f"初始化 RAG 系统失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if rag_system:
            rag_system.cleanup()

if __name__ == "__main__":
    main()
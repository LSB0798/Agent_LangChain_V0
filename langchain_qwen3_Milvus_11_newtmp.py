"""
def _split_documents1(self, documents): 之前的分块方式
def _split_documents(self, documents):  优化大文件分块策略（修复类型不一致问题）
"""
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
    EMBEDDING_MODEL_NAME = "Qwens/Qwen3-Embedding-0.6B"  # 使用Qwen3嵌入模型
    RERANKER_MODEL_NAME = "Qwens/Qwen3-Reranker-0.6B"    # Reranker模型
    LLM_MODEL_NAME = "Qwen3-30B-A3B"  # LLM模型

    # 检索相关配置
    COLLECTION_NAME = "rag_collection"  # 集合名称
    # COLLECTION_NAME = "rag_collection_updated"  # 添加后缀
    MEMORY_COLLECTION_NAME = "memory_collection"  # 记忆集合名称
    INITIAL_RETRIEVAL_K = 20           # 初始检索数量
    RERANKER_TOP_K = 5                 # 重排序后返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）
    
    # Memory相关配置
    MEMORY_FILE = "conversation_memory.json"  # 记忆存储文件
    MAX_CONVERSATION_TURNS = 10  # 最大对话轮次记忆
    MAX_SUMMARY_LENGTH = 500     # 摘要最大长度
    MEMORY_RETRIEVAL_WEIGHT = 0.3  # 记忆检索权重
    ENABLE_MEMORY_SUMMARY = True   # 启用记忆摘要

    client = OpenAI(base_url="http://-.-.-.-:-/v1", api_key='EMPTY')

    # 新增语言检测配置
    LANG_DETECTION_THRESHOLD = 0.5  # 语言检测置信度阈值
    ENGLISH_CHUNK_SIZE = 1500       # 英文为主文档的块大小
    ENGLISH_CHUNK_OVERLAP = 80     # 英文为主文档的块重叠
    MIXED_CHUNK_SIZE = 1200         # 混合文档的块大小
    MIXED_CHUNK_OVERLAP = 65       # 混合文档的块重叠

    # 新增记忆集合配置（从文档2添加）
    PROCEDURAL_MEMORY_COLLECTION = "procedural_memory"  # 长期偏好记忆
    EPISODIC_MEMORY_COLLECTION = "episodic_memory"      # 情景记忆（对话历史）
    SEMANTIC_MEMORY_COLLECTION = "semantic_memory"      # 语义记忆（事实知识）

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
    
    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self._check_embedding_dimension()  # 先检查维度
        self._init_memory_collections()
        print("记忆系统初始化完成（基于Milvus存储）")
    
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
            print("开始初始化记忆集合...")
            
            # 添加集合初始化间隔，避免资源竞争
            collections_info = [
                (self.config.PROCEDURAL_MEMORY_COLLECTION, "用户偏好与行为规则"),
                (self.config.EPISODIC_MEMORY_COLLECTION, "对话历史与事件记录"), 
                (self.config.SEMANTIC_MEMORY_COLLECTION, "事实性知识")
            ]
            
            for i, (name, desc) in enumerate(collections_info):
                print(f"\n初始化第{i+1}个集合: {name}")
                
                # 添加延迟，避免同时加载多个集合
                if i > 0:
                    time.sleep(2)
                    
                collection = self._get_or_create_collection(name, desc)
                
                if name == self.config.PROCEDURAL_MEMORY_COLLECTION:
                    self.procedural_memory = collection
                elif name == self.config.EPISODIC_MEMORY_COLLECTION:
                    self.episodic_memory = collection
                elif name == self.config.SEMANTIC_MEMORY_COLLECTION:
                    self.semantic_memory = collection
                    
                print(f"✅ 集合 {name} 初始化完成")
                
            print("所有记忆集合初始化完成")
            
        except Exception as e:
            print(f"初始化记忆集合失败: {e}")
            # 记录详细错误信息
            import traceback
            traceback.print_exc()
    
    def force_recreate_memory_collections(self):
        """强制重新创建所有记忆集合"""
        print("强制重新创建记忆集合...")
        
        collections_to_recreate = [
            self.config.PROCEDURAL_MEMORY_COLLECTION,
            self.config.EPISODIC_MEMORY_COLLECTION, 
            self.config.SEMANTIC_MEMORY_COLLECTION
        ]
        
        for collection_name in collections_to_recreate:
            try:
                if utility.has_collection(collection_name):
                    utility.drop_collection(collection_name)
                    print(f"已删除集合: {collection_name}")
            except Exception as e:
                print(f"删除集合 {collection_name} 失败: {e}")
        
        # 重新初始化
        self._init_memory_collections()
    
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
        """获取或创建记忆集合（强制重建问题集合）"""
        import time
        from pymilvus import MilvusException
        
        print(f"\n=== 开始处理集合: {name} ===")
        
        # 对于 episodic_memory 集合，强制删除并重建
        if name == self.config.EPISODIC_MEMORY_COLLECTION:
            print(f"⚠️ 强制删除并重建集合: {name}")
            try:
                if utility.has_collection(name):
                    utility.drop_collection(name)
                    print(f"已删除集合: {name}")
            except Exception as e:
                print(f"删除集合 {name} 失败: {e}")
        
        # 检查集合是否存在
        if utility.has_collection(name):
            try:
                print(f"集合 {name} 已存在，尝试加载...")
                collection = Collection(name)
                print(f"已获取集合对象: {name}")
                
                if self._check_dimension_match(collection):
                    print(f"维度匹配，开始加载集合 {name}...")
                    
                    # 添加超时和重试机制
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            print(f"第{attempt+1}次尝试加载集合 {name}...")
                            start_time = time.time()
                            
                            # 设置加载超时
                            collection.load(_timeout=30)
                            end_time = time.time()
                            
                            print(f"✅ 集合 {name} 加载成功，耗时: {end_time - start_time:.2f}秒")
                            print(f"集合 {name} 实体数量: {collection.num_entities}")
                            print(f"使用现有记忆集合: {name}")
                            return collection
                            
                        except MilvusException as e:
                            print(f"❌ 第{attempt+1}次加载集合 {name} 失败: {e}")
                            if attempt < max_retries - 1:
                                wait_time = 5
                                print(f"等待{wait_time}秒后重试...")
                                time.sleep(wait_time)
                                # 重新获取集合对象
                                collection = Collection(name)
                            else:
                                print(f"❌ 集合 {name} 加载失败，达到最大重试次数，删除并重建...")
                                utility.drop_collection(name)
                                break
                                
                        except Exception as e:
                            print(f"❌ 加载集合 {name} 时发生未知错误: {e}")
                            import traceback
                            traceback.print_exc()
                            if attempt == max_retries - 1:
                                utility.drop_collection(name)
                            break
                    
                    # 如果重试后仍然失败，创建新集合
                    print("加载失败，创建新集合...")
                    
                else:
                    print("❌ 维度不匹配，删除旧集合")
                    utility.drop_collection(name)
                    
            except Exception as e:
                print(f"❌ 处理现有集合时出错: {e}")
                import traceback
                traceback.print_exc()
                try:
                    utility.drop_collection(name)
                except:
                    pass
        
        # 创建新的记忆集合
        print(f"创建新集合: {name}")
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.actual_embedding_dim),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            
            schema = CollectionSchema(fields=fields, description=description)
            collection = Collection(name=name, schema=schema)
            print(f"集合 {name} Schema 创建成功")
            
            # 创建索引
            index_params = {
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 16}
            }
            
            collection.create_index(field_name="embedding", index_params=index_params)
            print(f"集合 {name} 嵌入索引创建成功")
            
            collection.create_index(field_name="user_id", index_params={"index_type": "TRIE"})
            print(f"集合 {name} 用户ID索引创建成功")
            
            # 加载新集合
            collection.load(_timeout=30)
            print(f"✅ 新集合 {name} 创建并加载成功")
            return collection
            
        except Exception as e:
            print(f"❌ 创建新集合 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
            # 清理可能创建失败的部分
            try:
                utility.drop_collection(name)
            except:
                pass
            raise e

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
        """添加对话轮次到记忆系统（修复版）"""
        try:
            # 存储用户查询到情景记忆
            self.store_memory(
                memory_type="episodic",
                content=f"用户问题: {query}",
                importance=0.7,  # 提高重要性权重
                metadata={
                    "type": "user_query",
                    "timestamp": datetime.now().isoformat(),
                    "context": context or []
                }
            )
            
            # 存储助手回答到情景记忆
            self.store_memory(
                memory_type="episodic", 
                content=f"助手回答: {response}",
                importance=0.6,
                metadata={
                    "type": "assistant_response",
                    "related_query": query,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # 同时存储完整的对话轮次
            self.store_memory(
                memory_type="episodic",
                content=f"用户: {query}\n助手: {response}",
                importance=0.8,  # 完整对话有更高重要性
                metadata={
                    "context": context or [],
                    "type": "conversation_turn",
                    "timestamp": datetime.now().isoformat(),
                    "query": query,  # 明确存储问题
                    "response": response  # 明确存储回答
                }
            )
            print(f"✅ 已存储对话轮次: Q: {query[:30]}... A: {response[:30]}...")
            
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
        """获取最近的对话历史（修复版）"""
        try:
            # 使用更精确的查询条件
            results = self.episodic_memory.query(
                expr='metadata["type"] == "conversation_turn"',
                output_fields=["content", "metadata", "created_at"],
                limit=limit,
                order_by="created_at desc"  # 按时间倒序
            )
            
            # 格式化结果
            conversation_history = []
            for result in results:
                metadata = result.get("metadata", {})
                content = result.get("content", "")
                
                # 直接从metadata中获取问题回答，避免解析错误
                if "query" in metadata and "response" in metadata:
                    conversation_history.append({
                        "query": metadata["query"],
                        "response": metadata["response"],
                        "timestamp": metadata.get("timestamp", ""),
                        "content": content
                    })
                elif "用户:" in content and "助手:" in content:
                    # 备用解析方法
                    parts = content.split("\n")
                    if len(parts) >= 2:
                        conversation_history.append({
                            "query": parts[0].replace("用户: ", ""),
                            "response": parts[1].replace("助手: ", ""),
                            "timestamp": metadata.get("timestamp", ""),
                            "content": content
                        })
            
            # 按时间正序排列（最早的在前）
            conversation_history.sort(key=lambda x: x.get("timestamp", ""))
            return conversation_history[-limit:]  # 返回最新的limit条
            
        except Exception as e:
            print(f"获取最近对话历史失败: {e}")
            return []

    def get_contextual_prompt(self, query: str) -> str:
        """构建包含记忆上下文的提示词（增强版）"""
        # 获取相关记忆
        relevant_memories = self.get_relevant_memories(query, top_k=3)
        
        # 获取最近对话（确保包含当前会话）
        recent_history = self.get_recent_conversation_history(limit=5)
        
        memory_context = ""
        if relevant_memories or recent_history:
            memory_context = "对话历史上下文：\n"
            
            # 添加相关记忆
            if relevant_memories:
                memory_context += "相关对话：\n"
                for i, mem in enumerate(relevant_memories, 1):
                    memory_context += f"{i}. 用户: {mem['query']}\n   回答: {mem['response']}\n"
            
            # 添加最近对话（避免重复）
            if recent_history:
                memory_context += "\n最近对话：\n"
                for i, turn in enumerate(recent_history, 1):
                    # 检查是否已经包含在相关记忆中
                    is_duplicate = any(
                        mem['query'] == turn['query'] and mem['response'] == turn['response'] 
                        for mem in (relevant_memories if relevant_memories else [])
                    )
                    if not is_duplicate:
                        memory_context += f"{i}. 用户: {turn['query']}\n   回答: {turn['response']}\n"
        
        # 用户偏好上下文保持不变
        user_preferences = self.retrieve_memories(
            "用户偏好", 
            memory_type="procedural", 
            top_k=2,
            min_importance=0.7
        )
        
        preference_context = ""
        if "procedural" in user_preferences:
            preference_context = "\n用户偏好：\n"
            for pref in user_preferences["procedural"]:
                preference_context += f"- {pref['content']}\n"
        
        return memory_context + preference_context

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
        
        # 初始化记忆系统
        self.memory = AgentMemorySystem(config, self.embeddings)

    def _start_milvus_lite(self):
        """直接使用外部 Milvus 服务"""
        try:
            # 直接连接外部 Milvus 服务，不启动内置的
            connections.connect(
                alias="default",
                host="localhost",
                port=19530
            )
            print("成功连接到外部 Milvus 服务器: localhost:19530")
        except Exception as e:
            print(f"连接外部 Milvus 服务器失败: {e}")
            raise Exception("无法连接到外部 Milvus 服务器，请确保服务已启动")

    def _connect_milvus(self):
        """连接到 Milvus 服务（简化版本）"""
        # 由于已经在 _start_milvus_lite 中连接，这里不需要重复连接
        print("Milvus 连接已在 _start_milvus_lite 中建立")
        return

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
        """加载已存在的 Milvus 集合或创建新的集合"""
        # 统一索引和搜索参数配置
        index_params = {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 16,
                "efConstruction": 200  # 构建时参数
            }
        }
        # 搜索时默认参数（确保ef大于INITIAL_RETRIEVAL_K）
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
                search_params=search_params  # 新增搜索默认参数
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
                search_params=search_params  # 新增搜索默认参数
            )

        split_docs = self._split_documents(documents)
        print(f"文档分块完成，共 {len(split_docs)} 个文本块")
        t1 = time.time()
        
        """vector_db = Milvus.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name=self.config.COLLECTION_NAME,
            connection_args={
                "host": "localhost",
                "port": default_server.listen_port
            },
            index_params=index_params,  # 创建时指定索引参数
            search_params=search_params  # 创建时指定默认搜索参数
        )
        print(f"Milvus 集合已创建: {self.config.COLLECTION_NAME}")"""
        # -------------------------------------------------------- #
        # 优化后：分批插入（每批1000个，代码中已支持 add_documents 方法）
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

    def answer_query(self, query: str, use_memory: bool = True) -> str:
        """根据查询生成回答（增强版，包含记忆功能）"""
        # 第一步：初始检索
        # search_params = {"ef": 150}  # ef应该大于k值
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
        
        # 第二步：重排序
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
            # prompt = f"""你是一个专业的AI助手，请严格遵循以下要求：

            #             信息来源约束：仅使用以下两部分信息回答问题，不得引入外部知识：
            #                 1. 对话历史信息：{memory_context}
            #                 2. RAG检索信息：{context}

            #             回答要求：
            #                 - 用中文专业、连贯地回答
            #                 - 如信息不足，明确说明"根据现有信息无法回答该问题"
            #                 - 如能回答，请注明主要信息来源（对话历史或RAG文档）

            #             用户问题：{query}

            #             请开始回答："""
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
                                        (2). 若能结合信息来源回应，需注明主要信息来源（对话历史或 RAG 文档）；若信息不足，明确说明 “根据现有信息无法回答该问题”。
                                        (3). 不得编造信息，严格基于给定的对话历史和 RAG 检索信息展开回应。
                                # 限制:
                                        (1). 不评判、不指责用户的情绪和行为，不输出任何负面、消极的话语，不传播焦虑或负能量。
                                        (2). 若用户询问非情感相关的专业问题（如学术、法律、医疗等），坦诚说明 “该问题属于专业领域范畴，我无法解答，建议咨询相关专业人士”，不编造答案。
                                        (3). 尊重用户边界，若用户表示不想多说，不追问，仅表达陪伴态度（如 “没关系，等你想聊的时候，我一直都在”）。"""
            USER_PROMPT = f""" # 信息来源约束:
                                仅使用以下两部分信息回应，不得引入外部知识：
                                1, 对话历史信息：{memory_context}
                                2, RAG 检索信息：{context}
                          # 用户问题：{query}
                          # 请开始回应："""

        # print('提示词: {}'.format(prompt[:500] + '...' if len(prompt) > 500 else prompt))
        print('===========================================')

        # 生成回答
        # response = self.llm(prompt)
        # response = response[len(prompt):]
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

        # 保存到记忆
        if use_memory:
            self.memory.add_conversation_turn(query, response, final_docs)
            # 同时提取用户偏好
            self.memory.extract_user_preference(query, response)
        
        return response

    def show_conversation_history(self):
        """显示对话历史"""
        if not self.memory.conversation_history:
            print("暂无对话历史")
            return
            
        print("\n=== 对话历史 ===")
        for i, turn in enumerate(self.memory.conversation_history, 1):
            print(f"\n第{i}轮对话 ({turn['timestamp']}):")
            print(f"用户: {turn['query']}")
            print(f"系统: {turn['response'][:200]}...")
    
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

        # 添加强制重建选项（临时调试用）
        print('-0-' * 10)
        rag_system.memory.force_recreate_memory_collections()
        print('-1-' * 10)
        
        print("\n增强版 RAG 问答系统已启动（包含专业Memory功能）")
        print("可用命令:")
        print("  - 直接输入问题进行查询")
        print("  - 'history': 查看对话历史")
        print("  - 'clear': 清空对话记忆")
        print("  - 'quit': 退出系统")
        print("  - 'update_milvus': 更新Milvus-RAG")
        
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
            elif user_input.lower() == 'update_milvus':
                rag_system.recreate_collection()
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

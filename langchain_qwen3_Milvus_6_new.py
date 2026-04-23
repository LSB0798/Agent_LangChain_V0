
import os
import torch
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer as RerankerTokenizer

# 引入 Milvus Lite 相关组件
from milvus import default_server
from langchain_community.vectorstores import Milvus
from pymilvus import connections, utility

class Config:
    # 文档相关配置
    DOCUMENTS_DIR = "documents"  # 本地文档目录（需手动创建）
    CHUNK_SIZE = 500             # 每个文本块的字符数（中文适配）
    CHUNK_OVERLAP = 50           # 块间重叠字符数（避免分割丢失上下文）
    
    # 模型相关配置
    EMBEDDING_MODEL_NAME = "/data/lishuaibing/Qwens/Qwen3-Embedding-0.6B"  # 使用Qwen3嵌入模型
    RERANKER_MODEL_NAME = "/data/lishuaibing/Qwens/Qwen3-Reranker-0.6B"    # Reranker模型
    LLM_MODEL_NAME = "/data/lishuaibing/Qwen3-30B-A3B"  # LLM模型

    # 检索相关配置
    COLLECTION_NAME = "rag_collection"  # 集合名称
    # COLLECTION_NAME = "rag_collection_updated"  # 添加后缀
    MEMORY_COLLECTION_NAME = "memory_collection"  # 记忆集合名称
    INITIAL_RETRIEVAL_K = 10           # 初始检索数量
    RERANKER_TOP_K = 5                 # 重排序后返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）
    
    # Memory相关配置
    MEMORY_FILE = "conversation_memory.json"  # 记忆存储文件
    MAX_CONVERSATION_TURNS = 10  # 最大对话轮次记忆
    MAX_SUMMARY_LENGTH = 500     # 摘要最大长度
    MEMORY_RETRIEVAL_WEIGHT = 0.3  # 记忆检索权重
    ENABLE_MEMORY_SUMMARY = True   # 启用记忆摘要

    # 新增 Milvus 配置
    MILVUS_PORT = 19530  # 指定固定端口
    MILVUS_DATA_DIR = "/data/lishuaibing/milvus_data"  # 数据目录
    MILVUS_START_TIMEOUT = 300  # 启动超时时间（秒）

class ConversationMemory:
    """对话记忆管理类"""
    
    def __init__(self, config, embeddings, vector_db=None):
        self.config = config
        self.embeddings = embeddings
        self.vector_db = vector_db
        """
        存储最近N轮对话, 
            config.MAX_CONVERSATION_TURNS = 10 
            表示最多存储10轮对话, 使用 deque 数据结构，
            当超过10轮时会自动移除最早的对话, 
        """
        self.conversation_history = deque(maxlen=config.MAX_CONVERSATION_TURNS)
        self.memory_summary = ""
        self.load_memory()
        
    def add_conversation_turn(self, query: str, response: str, context: List[str] = None):
        """添加对话轮次到记忆"""
        turn = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context": context or [],
            "embedding": None
        }
        
        # 生成嵌入（用于后续检索）
        try:
            """
            记忆向量表示, 
                为每个对话轮次生成向量嵌入表示, 
                使用嵌入模型将"查询+回答"的文本转换为向量, 
                存储在 turn["embedding"] 字段中，用于后续的语义检索
            """
            turn["embedding"] = self.embeddings.embed_query(query + " " + response)
        except Exception as e:
            print(f"生成记忆嵌入时出错: {e}")
            
        self.conversation_history.append(turn)
        self._update_memory_summary()
        self.save_memory()
        
    def get_relevant_memories(self, query: str, top_k: int = 3) -> List[Dict]:
        """获取与当前查询相关的历史记忆"""
        if not self.conversation_history:
            return []
            
        # 计算查询与历史记忆的相似度
        similarities = []
        query_embedding = self.embeddings.embed_query(query)
        
        for i, turn in enumerate(self.conversation_history):
            if turn["embedding"] is not None:
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, turn["embedding"])
                # 时间衰减因子（越近的记忆权重越高）
                time_factor = 1.0 - (i / len(self.conversation_history)) * 0.3
                weighted_similarity = similarity * time_factor
                similarities.append((weighted_similarity, turn))
        
        # 按相似度排序并返回top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in similarities[:top_k]]
    
    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        vec1 = torch.tensor(vec1)
        vec2 = torch.tensor(vec2)
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    def _update_memory_summary(self):
        """更新记忆摘要"""
        if not self.config.ENABLE_MEMORY_SUMMARY or not self.conversation_history:
            return
            
        # 简单的摘要生成：取最近几轮对话的关键信息
        recent_turns = list(self.conversation_history)[-3:]  # 最近3轮
        summary_parts = []
        
        for turn in recent_turns:
            # 提取关键信息（简化处理）
            summary_parts.append(f"用户问: {turn['query'][:100]}...")
            summary_parts.append(f"回答: {turn['response'][:100]}...")
        
        """
        对话摘要:
            对话摘要存储在 self.memory_summary 中
            最大长度为 config.MAX_SUMMARY_LENGTH = 500 字符
            包含最近3轮对话的关键信息
        """
        self.memory_summary = "。".join(summary_parts)[:self.config.MAX_SUMMARY_LENGTH]
    
    def get_contextual_prompt(self, query: str) -> str:
        """构建包含记忆上下文的提示词"""
        relevant_memories = self.get_relevant_memories(query)
        
        memory_context = ""
        if relevant_memories:
            memory_context = "之前的相关对话：\n"
            for i, mem in enumerate(relevant_memories, 1):
                memory_context += f"{i}. 用户: {mem['query']}\n   回答: {mem['response']}\n"
        
        summary_context = f"\n对话摘要: {self.memory_summary}" if self.memory_summary else ""
        
        return memory_context + summary_context
    
    def save_memory(self):
        """保存记忆到文件"""
        try:
            memory_data = {
                "conversation_history": list(self.conversation_history),
                "memory_summary": self.memory_summary,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.config.MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存记忆时出错: {e}")
    
    def load_memory(self):
        """从文件加载记忆"""
        try:
            if os.path.exists(self.config.MEMORY_FILE):
                with open(self.config.MEMORY_FILE, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    
                self.conversation_history = deque(
                    memory_data.get("conversation_history", []), 
                    maxlen=self.config.MAX_CONVERSATION_TURNS
                )
                self.memory_summary = memory_data.get("memory_summary", "")
                print(f"已加载 {len(self.conversation_history)} 轮历史对话记忆")
                
        except Exception as e:
            print(f"加载记忆时出错: {e}")
    
    def clear_memory(self):
        """清空记忆"""
        self.conversation_history.clear()
        self.memory_summary = ""
        if os.path.exists(self.config.MEMORY_FILE):
            os.remove(self.config.MEMORY_FILE)
        print("记忆已清空")

class EnhancedRAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.reranker_model, self.reranker_tokenizer = self._init_reranker()
        self.llm = self._init_llm()

        # 启动 Milvus Lite 服务器
        # actual_port = self._start_milvus_lite()
        
        # 连接 Milvus
        # connected_port = self._connect_milvus(actual_port)
        
        # 加载或创建向量数据库 - 修复参数问题
        # self.vector_db = self._load_or_create_vector_db(port=connected_port)  # 使用命名参数
        
        # 初始化记忆系统
        # self.memory = ConversationMemory(config, self.embeddings, self.vector_db)



        # 启动 Milvus Lite 服务器
        actual_port = self._start_milvus_lite()
        
        # 连接 Milvus
        connected_port = self._connect_milvus(actual_port)
        self.milvus_port = connected_port  # 保存端口号
        
        # 加载或创建向量数据库
        self.vector_db = self._load_or_create_vector_db(port=self.milvus_port)
        
        # 初始化记忆系统
        self.memory = ConversationMemory(config, self.embeddings, self.vector_db)
        
    def _start_milvus_lite(self):
        """启动 Milvus Lite 服务器 - 修复版"""
        try:
            # 创建数据目录
            os.makedirs(self.config.MILVUS_DATA_DIR, exist_ok=True)
            print(f"创建数据目录: {self.config.MILVUS_DATA_DIR}")
            
            # 使用环境变量设置配置
            os.environ['MILVUS_DATA'] = self.config.MILVUS_DATA_DIR
            os.environ['MILVUS_PORT'] = str(self.config.MILVUS_PORT)
            os.environ['MILVUS_GRACEFUL_TIME'] = str(self.config.MILVUS_START_TIMEOUT)
            
            # 直接启动服务器
            default_server.start()
            actual_port = default_server.listen_port
            print(f"✅ Milvus Lite 启动成功，端口: {actual_port}")
            print(f"数据存储路径: {self.config.MILVUS_DATA_DIR}")
            
            # 等待服务器完全启动
            self._wait_for_milvus_ready(actual_port)
            return actual_port
            
        except Exception as e:
            print(f"❌ Milvus Lite 启动异常: {e}")
            # 尝试获取端口信息
            try:
                actual_port = default_server.listen_port
                print(f"尝试使用端口: {actual_port}")
                return actual_port
            except:
                print("无法获取端口信息，使用配置端口")
                return self.config.MILVUS_PORT

    def _wait_for_milvus_ready(self, port, timeout=60):
        """等待 Milvus 服务器完全就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 尝试连接验证服务状态
                connections.connect(
                    alias="health_check",
                    host="localhost",
                    port=port
                )
                connections.disconnect("health_check")
                print("✅ Milvus 服务器就绪")
                return True
            except Exception as e:
                print(f"⏳ 等待 Milvus 服务器启动... ({int(time.time() - start_time)}秒)")
                time.sleep(5)
        
        raise TimeoutError(f"Milvus 服务器启动超时（{timeout}秒）")

    def _connect_milvus(self, port=None):
        """连接到 Milvus Lite 服务 - 修复版"""
        max_retries = 10  # 增加重试次数
        retry_delay = 5   # 增加重试延迟
        
        # 确定要连接的端口
        if port is None:
            port = self.config.MILVUS_PORT
        
        for attempt in range(max_retries):
            try:
                print(f"尝试连接 Milvus (端口: {port}, 尝试 {attempt+1}/{max_retries})...")
                
                # 先断开可能存在的旧连接
                try:
                    connections.disconnect("default")
                except:
                    pass
                
                connections.connect(
                    alias="default",
                    host="localhost",
                    port=port
                )
                print(f"✅ 成功连接到 Milvus Lite: localhost:{port}")
                return port
                
            except Exception as e:
                print(f"连接失败: {e}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print(f"端口 {port} 的所有尝试均失败")
        
        # 所有尝试都失败
        raise ConnectionError("无法连接到 Milvus 服务")

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
                    normalize_embeddings=True
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
            max_new_tokens=8192,
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

    def _split_documents(self, documents):
        """将文档分块处理"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def _load_or_create_vector_db(self, port):
        """加载或创建向量数据库 - 修复版"""
        # 确保连接有效
        try:
            connections.connect(
                alias="temp_connection",
                host="localhost",
                port=port
            )
            connections.disconnect("temp_connection")
        except Exception as e:
            print(f"无法连接到 Milvus: {e}")
            raise ConnectionError(f"无法连接到 Milvus: {e}")

        # 检查集合是否存在
        if utility.has_collection(self.config.COLLECTION_NAME):
            print(f"加载现有 Milvus 集合: {self.config.COLLECTION_NAME}")
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost",
                    "port": port  # 使用传入的端口
                },
            )

        # 新建集合
        print("创建新的 Milvus 集合...")
        documents = self._load_documents()
        if not documents:
            print("没有加载到文档，创建空集合")
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost",
                    "port": port  # 使用传入的端口
                },
            )

        split_docs = self._split_documents(documents)
        print(f"文档分块完成，共 {len(split_docs)} 个文本块")
        
        # 创建并插入数据到 Milvus
        vector_db = Milvus.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name=self.config.COLLECTION_NAME,
            connection_args={
                "host": "localhost",
                "port": port  # 使用传入的端口
            },
        )
        print(f"Milvus 集合已创建: {self.config.COLLECTION_NAME}")
        return vector_db
    
    def recreate_collection(self):
        """删除并重新创建向量集合（用于更新文档）"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.config.COLLECTION_NAME):
                # 删除现有集合
                utility.drop_collection(self.config.COLLECTION_NAME)
                print(f"已删除集合: {self.config.COLLECTION_NAME}")
            
            # 重新创建集合（使用保存的端口号）
            self.vector_db = self._load_or_create_vector_db(port=self.milvus_port)
            print("集合已重新创建，文档已更新")
            
        except Exception as e:
            print(f"重新创建集合时出错: {e}")
    
    def recreate_collection1(self):
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
            prompt = f""" # 角色: 
                                你是一个温柔共情的情感陪护机器人，擅长倾听用户的情绪倾诉，用温暖、包容的话语给予理解、安慰与支持，帮用户缓解压力、疏导情绪，让用户感受到被重视、被接纳。
                          # 信息来源约束:
                                仅使用以下两部分信息回应，不得引入外部知识：
                                1, 对话历史信息：{memory_context}
                                2, RAG 检索信息：{context}
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
                                (3). 尊重用户边界，若用户表示不想多说，不追问，仅表达陪伴态度（如 “没关系，等你想聊的时候，我一直都在”）。
                          # 用户问题：{query}
                          # 请开始回应："""
        else:
            prompt = f"""基于以下上下文信息，用中文回答用户的问题。如果无法从上下文找到答案，请说明无法回答。

                        上下文信息:
                        {context}

                        用户问题: {query}

                        请给出你的回答:"""

        # print('提示词: {}'.format(prompt[:500] + '...' if len(prompt) > 500 else prompt))
        print('===========================================')

        # 生成回答
        print('len prompt : {}'.format(len(prompt)))
        response = self.llm(prompt)
        print('prompt : {}'.format(prompt))
        print('response : {}'.format(response))
        response = response[len(prompt):]
        # 保存到记忆
        if use_memory:
            self.memory.add_conversation_turn(query, response, final_docs)
        
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
            self.memory.save_memory()
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
import os
import torch
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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
    INITIAL_RETRIEVAL_K = 10           # 初始检索数量
    RERANKER_TOP_K = 3                 # 重排序后返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）
    
    # 新增：Milvus 配置
    MILVUS_PORT = 19530  # 指定固定端口
    MILVUS_START_TIMEOUT = 300  # 启动超时时间（秒）
    MILVUS_DATA_DIR = "/data/lishuaibing/milvus_data"  # 数据目录

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.reranker_model, self.reranker_tokenizer = self._init_reranker()
        self.llm = self._init_llm()
        
        # 启动 Milvus Lite 服务器
        self._start_milvus_lite()
        
        # 连接 Milvus
        self._connect_milvus()
        self.vector_db = self._load_or_create_vector_db()

    def _start_milvus_lite(self):
        """启动 Milvus Lite 服务器 - 增强版"""
        try:
            # 创建数据目录
            os.makedirs(self.config.MILVUS_DATA_DIR, exist_ok=True)
            
            # 设置端口和数据目录
            default_server.set_base_config("port", str(self.config.MILVUS_PORT))
            default_server.set_base_config("dataDir", self.config.MILVUS_DATA_DIR)
            default_server.set_base_config("gracefulTime", str(self.config.MILVUS_START_TIMEOUT))
            
            # 启动服务器
            default_server.start()
            actual_port = default_server.listen_port
            print(f"✅ Milvus Lite 启动成功，端口: {actual_port}")
            print(f"数据目录: {self.config.MILVUS_DATA_DIR}")
            
            # 等待服务器完全启动
            self._wait_for_milvus_start(actual_port)
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

    def _wait_for_milvus_start(self, port, timeout=60):
        """等待 Milvus 服务器完全启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 尝试连接以验证服务是否就绪
                connections.connect(
                    alias="health_check",
                    host="localhost",
                    port=port
                )
                connections.disconnect("health_check")
                print("✅ Milvus 服务器就绪")
                return True
            except:
                print("⏳ 等待 Milvus 服务器启动...")
                time.sleep(5)
        raise TimeoutError(f"Milvus 服务器启动超时（{timeout}秒）")

    def _connect_milvus(self):
        """连接到 Milvus Lite 服务 - 增强版"""
        max_retries = 10  # 增加重试次数
        retry_delay = 5   # 增加重试延迟
        
        # 尝试的端口列表（配置端口和可能的活动端口）
        ports_to_try = [self.config.MILVUS_PORT]
        try:
            ports_to_try.append(default_server.listen_port)
        except:
            pass
        
        for port in ports_to_try:
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
        
        raise ConnectionError("无法连接到 Milvus 服务")

    def _init_embeddings(self):
        """初始化 Qwen3 嵌入模型"""
        print(f"加载 Qwen3 嵌入模型: {self.config.EMBEDDING_MODEL_NAME}")
        
        # 使用SentenceTransformer加载嵌入模型
        embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL_NAME)
        
        # 创建自定义的嵌入函数以适配LangChain接口
        class Qwen3Embeddings:
            def __init__(self, model):
                self.model = model
                
            def embed_documents(self, texts):
                """为文档生成嵌入"""
                return self.model.encode(
                    texts,
                    normalize_embeddings=True
                ).tolist()
                
            def embed_query(self, text):
                """为查询生成嵌入（使用查询提示）"""
                return self.model.encode(
                    [text],
                    prompt_name="query",
                    normalize_embeddings=True
                ).tolist()[0]
        
        return Qwen3Embeddings(embedding_model)

    def _init_reranker(self):
        """初始化 Qwen3 Reranker 模型"""
        print(f"加载 Qwen3 Reranker 模型: {self.config.RERANKER_MODEL_NAME}")
        
        # 加载reranker模型和分词器
        reranker_tokenizer = RerankerTokenizer.from_pretrained(
            self.config.RERANKER_MODEL_NAME, 
            padding_side='left'
        )
        reranker_model = AutoModelForCausalLM.from_pretrained(
            self.config.RERANKER_MODEL_NAME
        ).eval()
        
        # 配置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reranker_model = reranker_model.to(device)
        
        # 配置reranker参数
        self.token_false_id = reranker_tokenizer.convert_tokens_to_ids('no')
        self.token_true_id = reranker_tokenizer.convert_tokens_to_ids('yes')
        self.max_reranker_length = 8192
        
        # 配置前缀和后缀
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
        
        # 将张量移动到模型所在的设备
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

        # 格式化输入对
        pairs = [self._format_instruction(task_instruction, query, doc) for doc in documents]

        # 处理输入并计算分数
        inputs = self._process_reranker_inputs(pairs)
        scores = self._compute_reranker_scores(inputs)

        # 组合文档与分数并排序
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores

    def _init_llm(self):
        """初始化 Qwen3 语言模型"""
        print(f"加载 Qwen3 模型: {self.config.LLM_MODEL_NAME}")
        
        # 量化配置
        quantization_config = None
        if self.config.USE_4BIT_QUANTIZATION:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        # 创建文本生成管道
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

        # 定义不同类型文档的加载器
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
            separators=["\n\n", "\n", "。", "，", " ", ""]  # 中文优化分隔符
        )
        return text_splitter.split_documents(documents)

    def _load_or_create_vector_db(self):
        """加载已存在的 Milvus 集合或创建新的集合"""
        # 检查集合是否存在
        if utility.has_collection(self.config.COLLECTION_NAME):
            print(f"加载现有 Milvus 集合: {self.config.COLLECTION_NAME}")
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost",
                    "port": self.config.MILVUS_PORT  # 使用配置端口
                },
            )

        # 新建集合
        print("创建新的 Milvus 集合...")
        documents = self._load_documents()
        if not documents:
            print("没有加载到文档，创建空集合")
            # 创建空集合
            return Milvus(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                connection_args={
                    "host": "localhost", 
                    "port": self.config.MILVUS_PORT  # 使用配置端口
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
                "port": self.config.MILVUS_PORT  # 使用配置端口
            },
        )
        print(f"Milvus 集合已创建: {self.config.COLLECTION_NAME}")
        return vector_db

    def answer_query(self, query):
        """根据查询生成回答（使用Embeddings + Rerankers）"""
        # 第一步：初始检索（使用嵌入模型）
        relevant_docs = self.vector_db.similarity_search(
            query, 
            k=self.config.INITIAL_RETRIEVAL_K
        )
        
        # 提取文档内容
        candidate_docs = [doc.page_content for doc in relevant_docs]
        print(f"初始检索到 {len(candidate_docs)} 个候选文档")
        
        # 第二步：使用Reranker进行重排序
        if len(candidate_docs) > 1:  # 只有在多个候选文档时才需要重排序
            print("正在进行重排序...")
            reranked_docs = self._rerank_documents(query, candidate_docs)
            # 选择重排序后的top-k个文档
            top_docs = reranked_docs[:self.config.RERANKER_TOP_K]
            final_docs = [doc for doc, score in top_docs]
            print(f"重排序完成，选择前 {len(final_docs)} 个最相关文档")
        else:
            final_docs = candidate_docs
        
        # 构建上下文
        context = "\n\n".join(final_docs)
        print('检索到的上下文: {}'.format(context))
        print('---------------------------------------------')

        # 构建提示词
        prompt = f"""基于以下上下文信息，用中文回答用户的问题。如果无法从上下文找到答案，请说明无法回答。 用户问题是: {query}, 请给出你的回答:"""

        print('let us print prompts : {}'.format(prompt))
        print('===========================================')
        # 生成回答
        response = self.llm(prompt)
        return response

    def cleanup(self):
        """清理资源"""
        print("清理资源...")
        try:
            default_server.stop()
            print("Milvus Lite 服务器已停止")
        except Exception as e:
            print(f"停止 Milvus Lite 服务器时出错: {e}")

def main():
    # 初始化配置和 RAG 系统
    config = Config()
    rag_system = None
    
    try:
        rag_system = RAGSystem(config)
        
        # 交互问答
        print("\nRAG 问答系统已启动（使用Qwen3 Embeddings + Rerankers）")
        print("输入问题进行查询（输入 'quit' 退出）")
        while True:
            query = input("\n请输入问题: ")
            if query.lower() == 'quit':
                break
            if not query.strip():
                continue
                
            try:
                answer = rag_system.answer_query(query)
                print('++++++++++++++++++++++++++++++++++++++++++++++++')
                print("\n回答:", answer)
            except Exception as e:
                print(f"处理查询时出错: {e}")
                
    except Exception as e:
        print(f"初始化 RAG 系统失败: {e}")
        
    finally:
        # 确保资源被清理
        if rag_system:
            rag_system.cleanup()

if __name__ == "__main__":
    main()
import os
import time
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

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
    EMBEDDING_MODEL_NAME = "/data/lishuaibing/BAAI/bge-m3/"  # 中文嵌入模型
    # LLM_MODEL_NAME = "/data/lishuaibing/Qwen2___5-7B-Instruct"  # LLM模型
    LLM_MODEL_NAME = "/data/lishuaibing/Qwen3-30B-A3B/"  # LLM模型

    # 检索相关配置
    # Milvus Lite 配置
    COLLECTION_NAME = "rag_collection"  # 集合名称
    TOP_K = 3                    # 检索返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）

    # 新增：Milvus 端口配置
    MILVUS_PORT = 19530  # 指定 Milvus 端口

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        
        # 启动 Milvus Lite 服务器
        self._start_milvus_lite()
        
        # 连接 Milvus
        self._connect_milvus()
        self.vector_db = self._load_or_create_vector_db()

    def _start_milvus_lite(self):
        """启动 Milvus Lite 服务器"""
        try:
            # 设置环境变量指定端口
            os.environ['MILVUS_PORT'] = str(self.config.MILVUS_PORT)

            # 直接启动服务器，不检查 started 属性
            default_server.start()
            print(f"Milvus Lite 启动成功，端口: {default_server.listen_port}")
        except Exception as e:
            # 如果服务器已经启动，可能会抛出异常，但我们继续尝试连接
            print(f"Milvus Lite 启动异常（可能已启动）: {e}")
            # 尝试获取端口信息
            try:
                print(f"尝试使用端口: {default_server.listen_port}")
            except:
                print("无法获取端口信息")

    def _connect_milvus(self):
        """连接到 Milvus Lite 服务"""
        max_retries = 5
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                connections.connect(
                    alias="default",
                    host="localhost",
                    port=self.config.MILVUS_PORT  # 使用配置的端口
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
        """初始化 BGE-M3 嵌入模型"""
        print(f"加载 BGE-M3 嵌入模型: {self.config.EMBEDDING_MODEL_NAME}")
        query_instruction = "为这个句子生成表示以用于检索相关文章："
        return HuggingFaceBgeEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            query_instruction=query_instruction
        )

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
        """加载或创建向量数据库"""
        # 确保连接有效
        try:
            connections.connect(
                alias="temp_connection",
                host="localhost",
                port=self.config.MILVUS_PORT
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
                    "port": self.config.MILVUS_PORT
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
                    "port": self.config.MILVUS_PORT
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
                "port": self.config.MILVUS_PORT
            },
        )
        print(f"Milvus 集合已创建: {self.config.COLLECTION_NAME}")
        return vector_db

    def answer_query(self, query):
        """根据查询生成回答"""
        # 检索相关文档
        relevant_docs = self.vector_db.similarity_search(query, k=self.config.TOP_K)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print('检索到的上下文: {}'.format(context))

        # 构建提示词
        prompt = f"""基于以下上下文信息，用中文回答用户的问题。如果无法从上下文找到答案，请说明无法回答。

                上下文:
                {context}

                用户问题: {query}

                回答:"""

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
        print("\nRAG 问答系统已启动，输入问题进行查询（输入 'quit' 退出）")
        while True:
            query = input("\n请输入问题: ")
            if query.lower() == 'quit':
                break
            if not query.strip():
                continue
                
            try:
                answer = rag_system.answer_query(query)
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
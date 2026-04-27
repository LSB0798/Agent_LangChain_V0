import os
import torch
# from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

class Config:
    # 文档相关配置
    DOCUMENTS_DIR = "documents"  # 本地文档目录（需手动创建）
    CHUNK_SIZE = 500             # 每个文本块的字符数（中文适配）
    CHUNK_OVERLAP = 50           # 块间重叠字符数（避免分割丢失上下文）
    
    # 模型相关配置
    EMBEDDING_MODEL_NAME = "/data/BAAI/bge-m3"  # 中文嵌入模型
    # EMBEDDING_MODEL_NAME = "/data/data/NLP/llm_team/lsb/Qwens/Qwen3-Embedding-0.6B"  # Qwen3-embedding-0.6B 嵌入模型, 可以跑通, 语义输出奇怪
    # EMBEDDING_MODEL_NAME = "/data/data/NLP/llm_team/lsb/Qwens/Qwen3-Embedding-4B"  # Qwen3-embedding-0.6B 嵌入模型, 可以跑通, 语义输出奇怪
    # LLM_MODEL_NAME = "/data/Qwen2___5-7B-Instruct"  # Qwen-2.5-7B模型, 可以跑通
    # LLM_MODEL_NAME = "/data/data/NLP/llm_team/lsb/Qwens/Qwen3-0___6B"  # Qwen-3-0.6B模型, 可以跑通, 意味着其他亦可以
    LLM_MODEL_NAME = "/data/Qwen3-30B-A3B"  # Qwen-3-0.6B模型, 可以跑通, 意味着其他亦可以
    
    # 检索相关配置
    VECTOR_DB_DIR = "vector_db_qwen_bge_m3"  # 向量库存储目录
    TOP_K = 3                                # 检索返回的相关片段数量
    
    # 量化配置
    USE_4BIT_QUANTIZATION = True  # 启用4位量化（降低显存占用）

class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_db = self._load_or_create_vector_db()

    def _init_embeddings(self):
        """初始化BGE-M3嵌入模型"""
        print(f"加载BGE-M3嵌入模型: {self.config.EMBEDDING_MODEL_NAME}")
        query_instruction = "为这个句子生成表示以用于检索相关文章："
        return HuggingFaceBgeEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            query_instruction=query_instruction
        )

    def _init_llm(self):
        """初始化Qwen3语言模型"""
        print(f"加载Qwen3模型: {self.config.LLM_MODEL_NAME}")
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
                print(f"加载{len(docs)}个{ext}文档")
            except Exception as e:
                print(f"加载{ext}文档出错: {e}")

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
        """加载已存在的向量库或创建新的向量库"""
        if os.path.exists(self.config.VECTOR_DB_DIR):
            print(f"加载现有向量库: {self.config.VECTOR_DB_DIR}")
            return Chroma(
                persist_directory=self.config.VECTOR_DB_DIR,
                embedding_function=self.embeddings
            )

        # 新建向量库
        print("创建新的向量库...")
        documents = self._load_documents()
        if not documents:
            print("没有加载到文档，创建空向量库")
            return Chroma(
                persist_directory=self.config.VECTOR_DB_DIR,
                embedding_function=self.embeddings
            )

        split_docs = self._split_documents(documents)
        print(f"文档分块完成，共{len(split_docs)}个文本块")
        
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.config.VECTOR_DB_DIR
        )
        vector_db.persist()
        print(f"向量库已保存至: {self.config.VECTOR_DB_DIR}")
        return vector_db

    def answer_query(self, query):
        """根据查询生成回答"""
        # 检索相关文档
        relevant_docs = self.vector_db.similarity_search(query, k=self.config.TOP_K)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        print('context : {}'.format(context))

        # 构建提示词
        prompt = f"""基于以下上下文信息，用中文回答用户的问题。如果无法从上下文找到答案，请说明无法回答。

                上下文:
                {context}

                用户问题: {query}

                回答:"""

        # 生成回答
        response = self.llm(prompt)
        return response

def main():
    # 初始化配置和RAG系统
    config = Config()
    rag_system = RAGSystem(config)
    
    # 交互问答
    print("\nRAG问答系统已启动，输入问题进行查询（输入'quit'退出）")
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

if __name__ == "__main__":
    main()

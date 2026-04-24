"""
langchain_qwen3_Milvus_Qwen3_Embeddings_Rerankers.py
"""
from glob import glob
text_lines = []


for file_path in glob("milvus_docs/en/faq/*.md", recursive=True):
    with open(file_path, "r") as file:
        file_text = file.read()
        # print('file_text : {}'.format(file_text))
        # text_lines += file_text.split(<span class="hljs-string">&quot;# &quot;</span>)
        # 按 "# " 分割文本，并过滤空字符串（去除前后空白后仍为空的内容）
        text_lines += [s.strip() for s in file_text.split("# ") if s.strip()]
print('text_lines : {}'.format(text_lines))

# ---------------------------------------------------------------------------- #
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
print('-0-' * 10)
# Load Qwen3-Embedding-0.6B model for text embeddings
embedding_model = SentenceTransformer("/data/Qwens/Qwen3-Embedding-0.6B")
print('-1-' * 10)
# Load Qwen3-Reranker-0.6B model for reranking
reranker_tokenizer = AutoTokenizer.from_pretrained("/data/Qwens/Qwen3-Reranker-0.6B", padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained("/data/Qwens/Qwen3-Reranker-0.6B").eval()
print('-2-' * 10)
# Reranker configuration
token_false_id = reranker_tokenizer.convert_tokens_to_ids('no')
token_true_id = reranker_tokenizer.convert_tokens_to_ids('yes')
max_reranker_length = 8192
print('-3-' * 10)

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = '<|im_end|>\n<|im_start|>assistant\n\n\n\n\n'
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
print('-4-' * 10)

# ================================================================================================================================= #
# ------------------------------------------------------------ 嵌入功能 ------------------------------------------------------------ #
# ================================================================================================================================= #
def emb_text(text, is_query=False, normalize_embeddings=True):
    """
    Generate text embeddings using Qwen3-Embedding-0.6B model with enhanced robustness.
    
    Args:
        text (Union[str, List[str]]): Input text(s) to embed. Can be a single string or list of strings.
        is_query (bool): Whether the input is a query (True) or document (False). 
                        Uses "query" prompt for queries to improve retrieval performance.
        normalize_embeddings (bool): Whether to L2-normalize the embeddings. Defaults to True.
    
    Returns:
        Union[List[float], List[List[float]]]: Embedding(s) of the input text(s). 
                                            Returns single list if input is string, list of lists if input is list.
    
    Raises:
        ValueError: If input text is empty or not a valid string/list of strings.
        RuntimeError: If embedding generation fails due to model issues.
    """
    # Input validation
    if not text:
        raise ValueError("Input text cannot be empty")
    
    # Normalize input to list format for consistent processing
    if isinstance(text, str):
        texts = [text]
        is_single = True
    elif isinstance(text, list):
        if not all(isinstance(t, str) for t in text):
            raise ValueError("All elements in input list must be strings")
        texts = text
        is_single = False
    else:
        raise ValueError("Input text must be a string or list of strings")
    
    try:
        # Generate embeddings with appropriate prompt
        if is_query:
            embeddings = embedding_model.encode(
                texts,
                prompt_name="query",
                normalize_embeddings=normalize_embeddings
            )
        else:
            embeddings = embedding_model.encode(
                texts,
                normalize_embeddings=normalize_embeddings
            )
        
        # Convert to list format and handle single/multiple inputs
        embeddings_list = embeddings.tolist()
        return embeddings_list[0] if is_single else embeddings_list
    
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}") from e
print('-5-' * 10)

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"First 10 values: {test_embedding[:10]}")
print('-6-' * 10)

# ================================================================================================================================= #
# --------------------------------------------------------- Reranker 实现 --------------------------------------------------------- #
# ================================================================================================================================= #
"""
Reranker 使用交叉编码器架构来评估查询-文档对。这比双编码器嵌入模型的计算成本更高，但能提供更细致的相关性评分。

下面是完整的 Rerankers 流程：
"""
def format_instruction(instruction, query, doc):
    """Format instruction for reranker input"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output

def process_inputs_0(pairs):
    """Process inputs for reranker"""
    inputs = reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_reranker_length)
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    return inputs

def process_inputs(pairs):
    """处理输入对为模型可接受的张量格式"""
    # 使用tokenizer直接处理整个批次，避免手动操作
    inputs = reranker_tokenizer(
        pairs,
        padding=True,           # 启用填充
        truncation=True,        # 启用截断
        max_length=max_reranker_length,  # 设置最大长度
        return_tensors="pt",    # 返回PyTorch张量
        return_attention_mask=True)
    
    # 将张量移动到模型所在的设备
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    
    return inputs

@torch.no_grad()
def compute_logits(inputs): # , kwargs
    """Compute relevance scores using reranker"""
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def rerank_documents(query, documents, task_instruction=None):
    """
    Rerank documents based on query relevance using Qwen3-Reranker

    Args:
        query: Search query
        documents: List of documents to rerank
        task_instruction: Task instruction for reranking

    Returns:
        List of (document, score) tuples sorted by relevance score
    """
    if not documents:  # 处理空文档列表情况
        return []
        
    # 设置默认任务指令
    if task_instruction is None:
        task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'

    # 格式化输入对（指令+查询+文档）
    pairs = [format_instruction(task_instruction, query, doc) for doc in documents]

    # 处理输入，转换为模型可接受的格式
    inputs = process_inputs(pairs)

    # 计算相关性分数
    scores = compute_logits(inputs)

    # 组合文档与分数，并按分数降序排序
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    return doc_scores

"""
设置 Milvus 向量数据库
    现在让我们建立向量数据库。为了简单起见，我们使用 Milvus Lite，但同样的代码也适用于完整的 Milvus 部署：
"""

from pymilvus import MilvusClient
milvus_client = MilvusClient(uri="./milvus_demo.db")
print('-7-' * 10)

collection_name = "my_rag_collection"
print('-8-' * 10)

"""
部署选项：

    本地文件（如./milvus.db ）：使用 Milvus Lite，非常适合开发

    Docker/Kubernetes：使用服务器 URI，如http://localhost:19530 用于生产

    Zilliz Cloud：使用云端点和 API 密钥管理服务
"""
# 清理任何现有的 Collections 并创建一个新的：
# pip install "pymilvus[milvus_lite]>=2.4.2"

# Remove existing collection if it exists
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
# Create new collection with our embedding dimensions
milvus_client.create_collection(collection_name=collection_name,
                                dimension=embedding_dim,  # 1024 for Qwen3-Embedding-0.6B
                                metric_type="IP",  # Inner product for similarity
                                consistency_level="Strong",  # Ensure data consistency
                                )
print('-9-' * 10)

# 将数据加载到 Milvus 中
# 现在让我们处理我们的文档并将其插入向量数据库:

from tqdm import tqdm
data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
print('-10-' * 10)

"""
利用 Rerankers 技术增强 RAG
现在到了激动人心的部分--将这一切整合到一个完整的检索增强生成系统中。

步骤 1：查询和初始检索
让我们用一个关于 Milvus 的常见问题进行测试：
"""

question = "How is data stored in milvus?"
# Perform initial dense retrieval to get top candidates
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[emb_text(question, is_query=True)],  # Use query prompt
    limit=10,  # Get top 10 candidates for reranking
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["text"],  # Return the actual text content
)

print(f"Found {len(search_res[0])} initial candidates")
print('-11-' * 10)

"""
步骤 2：重排序以提高精确度
提取候选文档并应用 Rerankers：
"""
# Extract candidate documents
candidate_docs = [res["entity"]["text"] for res in search_res[0]]
# Rerank using Qwen3-Reranker
print("Reranking documents...")
reranked_docs = rerank_documents(question, candidate_docs)

# Select top 3 after reranking
top_reranked_docs = reranked_docs[:3]
print(f"Selected top {len(top_reranked_docs)} documents after reranking")

"""
步骤 3：比较结果
让我们来看看 Rerankers 如何改变结果：
"""
"""
Reranked results (top 3):
[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###",
        0.9997891783714294
    ],
    [
        "How does Milvus flush data?\n\nMilvus returns success when inserted data are loaded to the message queue. However, the data are not yet flushed to the disk. Then Milvus' data node writes the data in the message queue to persistent storage as incremental logs. If `flush()` is called, the data node is forced to write all data in the message queue to persistent storage immediately.\n\n###",
        0.9989748001098633
    ],
    [
        "Does the query perform in memory? What are incremental data and historical data?\n\nYes. When a query request comes, Milvus searches both incremental data and historical data by loading them into memory. Incremental data are in the growing segments, which are buffered in memory before they reach the threshold to be persisted in storage engine, while historical data are from the sealed segments that are stored in the object storage. Incremental data and historical data together constitute the whole dataset to search.\n\n###",
        0.9984032511711121
    ]
]
================================================================================
Original embedding-based results (top 3):
[
    [
        " Where does Milvus store data?\n\nMilvus deals with two types of data, inserted data and metadata. \n\nInserted data, including vector data, scalar data, and collection-specific schema, are stored in persistent storage as incremental log. Milvus supports multiple object storage backends, including [MinIO](https://min.io/), [AWS S3](https://aws.amazon.com/s3/?nc1=h_ls), [Google Cloud Storage](https://cloud.google.com/storage?hl=en#object-storage-for-companies-of-all-sizes) (GCS), [Azure Blob Storage](https://azure.microsoft.com/en-us/products/storage/blobs), [Alibaba Cloud OSS](https://www.alibabacloud.com/product/object-storage-service), and [Tencent Cloud Object Storage](https://www.tencentcloud.com/products/cos) (COS).\n\nMetadata are generated within Milvus. Each Milvus module has its own metadata that are stored in etcd.\n\n###", 
        0.8306853175163269
    ],
    [
        "How does Milvus flush data?\n\nMilvus returns success when inserted data are loaded to the message queue. However, the data are not yet flushed to the disk. Then Milvus’ data node writes the data in the message queue to persistent storage as incremental logs. If <span class="hljs-title">flush</span>() is called, the data node is forced to write all data in the message queue to persistent storage immediately.\n\n###", 
        0.7302717566490173
    ],
    [
        "How does Milvus handle vector data types and precision?\n\nMilvus supports Binary, Float32, Float16, and BFloat16 vector types.\n\n- Binary vectors: Store binary data as sequences of 0s and 1s, used in image processing and information retrieval.\n- Float32 vectors: Default storage with a precision of about 7 decimal digits. Even Float64 values are stored with Float32 precision, leading to potential precision loss upon retrieval.\n- Float16 and BFloat16 vectors: Offer reduced precision and memory usage. Float16 is suitable for applications with limited bandwidth and storage, while BFloat16 balances range and efficiency, commonly used in deep learning to reduce computational requirements without significantly impacting accuracy.\n\n###", 
        0.7003671526908875
    ]
]
"""

"""
与嵌入相似度得分相比，重排通常会显示出更高的判别得分（相关文档更接近 1.0）。

步骤 4：生成最终响应
    现在，让我们利用检索到的上下文生成一个综合答案：

    首先：将检索到的文档转换为字符串格式。
"""

# context = "\n".join([] for line_with_distance in retrieved_lines_with_distances])
# 从重排序后的文档中提取文本内容（忽略分数），并用换行符连接
context = "\n".join([doc for doc, score in top_reranked_docs])

"""
为大语言模型提供系统提示和用户提示。该提示由从 Milvus 检索到的文档生成。
"""

SYSTEM_PROMPT = """Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided."""
USER_PROMPT = f"""Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
                    <context>
                    {context}
                    </context>
                    <question>
                    {question}
                    </question>"""

from openai import OpenAI

client = OpenAI(base_url="http://10.20.223.89:61253/v1", api_key='EMPTY')

stream = client.chat.completions.create(
    model="qwen3-moe",
    messages=[{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": USER_PROMPT},
              ],
    stream=True,  # 启用流式输出
    max_tokens=4000
)
"""
response = openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)
print(response.choices[0].message.content)
"""

print("开始接收流式响应：")
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

"""
预期输出：

In Milvus, data is stored in two main forms: inserted data and metadata. 
Inserted data, which includes vector data, scalar data, and collection-specific 
schema, is stored in persistent storage as incremental logs. Milvus supports 
multiple object storage backends for this purpose, including MinIO, AWS S3, 
Google Cloud Storage, Azure Blob Storage, Alibaba Cloud OSS, and Tencent 
Cloud Object Storage. Metadata for Milvus is generated by its various modules 
and stored in etcd.

总结
本教程使用 Qwen3 的 embedding 和 rerankers 模型演示了完整的 RAG 实现。主要收获

两阶段检索（密集 + Rerankers）比纯嵌入方法持续提高准确性

通过指令提示，无需重新训练即可进行特定领域的调整

多语言功能自然运行，无需增加复杂性

可使用 0.6B 模型进行本地部署

Qwen3 系列以轻量级开源软件包的形式提供稳定的性能。虽然不是革命性的，但它们提供了渐进的改进和有用的功能，如指令提示，可以在生产系统中发挥真正的作用。

请根据您的具体数据和用例测试这些模型--哪种方法最有效始终取决于您的内容、查询模式和性能要求。
"""

import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PDFReader

# 加载PDF文件
parser = PDFReader()
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
   "llama-index/pdf",  # 确保这个路径存在且包含PDF文件
   file_extractor=file_extractor
).load_data()

# 设置嵌入模型（使用你本地的llama3:latest）
Settings.embed_model = OllamaEmbedding(model_name="llama3")

# 设置LLM模型（使用你本地的llama3:latest）
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# 检查是否已有存储的索引
if not os.path.exists("llama-index/db/pdf-db"):
    # 创建新索引
    index = VectorStoreIndex.from_documents(documents)
    # 持久化存储索引
    index.storage_context.persist(persist_dir="llama-index/db/pdf-db")
else:
    # 加载已有索引
    storage_context = StorageContext.from_defaults(persist_dir="llama-index/db/pdf-db")
    index = load_index_from_storage(storage_context)

# 创建查询引擎
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query("如果我想获得澳洲500签证，TOEFL最少要考多少分？")
print(response)
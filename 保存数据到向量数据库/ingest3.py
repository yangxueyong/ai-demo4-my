import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 加载环境变量
load_dotenv()


# ====================== 1. 停用词配置 ======================
def load_stopwords(stopwords_path: str):
    """加载本地停用词文件"""
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        print(f"✅ 成功加载 {len(stopwords)} 个停用词")
        return stopwords
    except Exception as e:
        print(f"⚠️ 加载停用词失败: {str(e)}，将使用默认停用词")
        return {"的", "了", "是", "在", "和"}  # 基础中文停用词


# 配置停用词（替换为你的实际路径）
STOPWORDS_PATH = "/Users/yxy/work/python/ai-demo4-my/stop-words/cn_stopwords.txt"  # 建议使用清华大学停用词库
Settings.stopwords = load_stopwords(STOPWORDS_PATH)
Settings.tokenizer = None  # 禁用NLTK分词器

# ====================== 2. 模型配置 ======================
# 滑动窗口解析器
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# BGE嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="/Users/yxy/work/ai/model-new/bge-small-zh-v1.5",
    embed_batch_size=32
)


# ====================== 3. 文档处理核心逻辑 ======================
def ingest_documents(directory_path: str, collection_name: str):
    try:
        # 读取文档
        reader = SimpleDirectoryReader(
            input_dir=directory_path,
            recursive=True,
            required_exts=[".pdf", ".docx", ".pptx", ".txt", ".md"]
        )
        documents = reader.load_data()
        print(f"📄 成功加载 {len(documents)} 个文档")

        # 滑动窗口分割
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"✂️ 分割为 {len(nodes)} 个文本块（已自动应用停用词过滤）")

        # 连接ChromaDB
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        chroma_client.delete_collection(collection_name);
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_client.get_or_create_collection(collection_name)
        )

        # 构建索引
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=embed_model,
            show_progress=True
        )

        print(f"\n✅ 完成！已存储 {len(nodes)} 个节点到 '{collection_name}'")
        print(f"🔗 ChromaDB可视化: http://localhost:8000")

    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        raise


# ====================== 4. 主程序 ======================
if __name__ == "__main__":
    # 配置参数
    DOC_DIR = "/Users/yxy/work/python/ai-demo4-my/data"  # 你的文档目录
    COLLECTION_NAME = "knowledge_base_v2"  # ChromaDB集合名

    # 运行处理
    ingest_documents(DOC_DIR, COLLECTION_NAME)

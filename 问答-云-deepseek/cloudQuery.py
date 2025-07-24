import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.deepseek import DeepSeek
import chromadb
from rich.console import Console
from rich.prompt import Prompt
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    KeywordNodePostprocessor
)

# 初始化控制台
console = Console()

# 加载环境变量
load_dotenv()

# 修复HuggingFace Tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ====================== 1. 自定义中文提示模板 ======================
def messages_to_prompt_chinese(messages):
    """强制使用中文的自定义提示模板"""
    system_prompt = {
        "role": "system",
        "content": "你是一个专业的中文AI助手，必须使用简体中文回答所有问题。回答时需满足以下要求："
                   "\n1. 严格使用中文回答，禁止输出任何英文内容"
                   "\n2. 只基于提供的上下文信息回答，不添加无关信息"
                   "\n3. 如果问题涉及多个主题，请明确区分"
                   "\n4. 回答内容要简洁准确，避免重复"
    }

    # 将系统提示插入到消息开头
    if messages[0]["role"] != "system":
        messages = [system_prompt] + messages

    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += f"<|system|>\n{message['content']}</s>\n"
        elif message["role"] == "user":
            prompt += f"<|user|>\n{message['content']}</s>\n"
        elif message["role"] == "assistant":
            prompt += f"<|assistant|>\n{message['content']}</s>\n"

    # 确保最后以助理响应开头结束
    if not prompt.endswith("<|assistant|>\n"):
        prompt += "<|assistant|>\n"

    return prompt


# ====================== 2. 配置云端DeepSeek模型 ======================
def configure_deepseek_api():
    """配置云端DeepSeek API"""
    try:
        console.print("🔄 正在配置DeepSeek API...", style="yellow")

        # 从环境变量获取API密钥
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到DEEPSEEK_API_KEY环境变量")

        # 配置DeepSeek模型
        llm = DeepSeek(
            api_key=api_key,
            model="deepseek-chat",
            temperature=0.3,
            max_tokens=1024,
            messages_to_prompt=messages_to_prompt_chinese,
        )

        Settings.llm = llm
        console.print("✅ DeepSeek API配置成功", style="green")
        return llm
    except Exception as e:
        console.print(f"❌ API配置失败: {str(e)}", style="red")
        raise


# ====================== 3. 初始化问答引擎 ======================
def initialize_query_engine(collection_name: str):
    """初始化ChromaDB查询引擎"""
    try:
        console.print("🔄 正在连接向量数据库...", style="yellow")

        # 连接ChromaDB
        chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        chroma_collection = chroma_client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 创建索引
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model="local:/Users/yxy/work/ai/model-new/bge-small-zh-v1.5"
        )

        # 创建查询引擎
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            vector_store_query_mode="hybrid",
            alpha=0.6,
            response_mode="compact",
            streaming=True,
            verbose=False,
        )

        console.print("✅ 向量数据库连接成功", style="green")
        return query_engine
    except Exception as e:
        console.print(f"❌ 数据库连接失败: {str(e)}", style="red")
        raise


# ====================== 4. 交互式问答循环 ======================
def chat_loop(query_engine):
    """交互式问答循环"""
    console.print("\n💬 请输入您的问题（输入'退出'或'quit'结束）", style="bold blue")

    while True:
        try:
            # 获取用户输入
            query = Prompt.ask("❓ 问题")

            # 退出条件
            if query.lower() in ["退出", "quit", "exit"]:
                console.print("👋 再见！", style="bold green")
                break

            if not query.strip():
                console.print("⚠️ 请输入有效问题", style="yellow")
                continue

            # 执行查询
            console.print("🔄 正在思考...", style="yellow")
            console.print("\n📚 回答:", style="bold green")

            try:
                # 获取流式响应
                streaming_response = query_engine.query(query)

                # 处理响应
                full_response = ""
                for token in streaming_response.response_gen:
                    console.print(token, end="", style="green")
                    full_response += token
                    console.file.flush()

                console.print()  # 换行

                # 显示来源
                if hasattr(streaming_response, 'source_nodes'):
                    console.print("\n🔍 来源参考:", style="bold blue")
                    for i, node in enumerate(streaming_response.source_nodes, 1):
                        source_info = node.node.metadata.get('file_name', '未知文件')
                        console.print(f"{i}. {source_info} [相似度: {node.score:.3f}]")

            except Exception as e:
                console.print(f"\n❌ 处理响应时出错: {str(e)}", style="red")

        except KeyboardInterrupt:
            console.print("\n👋 手动中断，再见！", style="bold red")
            break
        except Exception as e:
            console.print(f"❌ 发生错误: {str(e)}", style="red")


# ====================== 5. 主程序 ======================
if __name__ == "__main__":
    # 配置参数
    COLLECTION_NAME = "knowledge_base_v2"

    try:
        # 初始化模型和引擎
        console.print("⚙️ 正在初始化系统...", style="bold blue")
        configure_deepseek_api()
        query_engine = initialize_query_engine(COLLECTION_NAME)

        # 预热模型
        console.print("🔥 预热API中...", style="yellow")
        query_engine.query("你好")

        # 启动问答
        console.print("🚀 系统准备就绪", style="bold green")
        chat_loop(query_engine)

    except Exception as e:
        console.print(f"\n❌ 系统初始化失败: {str(e)}", style="red")

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
import chromadb
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown

# 初始化控制台
console = Console()

# 加载环境变量
load_dotenv()


# ====================== 1. 配置本地DeepSeek模型 ======================
def load_local_deepseek(model_path: str):
    """加载本地DeepSeek模型 (GGUF量化格式)"""
    try:
        console.print("🔄 正在加载本地DeepSeek模型...", style="yellow")

        # 配置本地模型参数
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.3,
            max_new_tokens=512,  # 减少最大token数
            context_window=2048,
            model_kwargs={
                "n_gpu_layers": 40,  # 增加Metal加速层数
                "main_gpu": 0,
                "n_threads": 6,  # 明确设置线程数
                "n_batch": 512,  # 增加批处理大小
                "use_mmap": True,  # 启用内存映射
                "use_mlock": True  # 锁定内存防止交换
            },
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=False,
        )

        Settings.llm = llm
        console.print("✅ 本地DeepSeek模型加载成功", style="green")
        return llm
    except Exception as e:
        console.print(f"❌ 模型加载失败: {str(e)}", style="red")
        raise


# ====================== 2. 初始化问答引擎 ======================
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
        # query_engine = index.as_query_engine(
        #     similarity_top_k=2,
        #     response_mode="compact",
        #     verbose=True
        # )
        query_engine = index.as_query_engine(
            similarity_top_k=2,
            vector_store_query_mode="hybrid",  # 混合检索模式
            alpha=0.5,  # 平衡关键字和向量搜索
            response_mode="compact",
            streaming=True,
            verbose=False  # 关闭详细日志
        )

        console.print("✅ 向量数据库连接成功", style="green")
        return query_engine
    except Exception as e:
        console.print(f"❌ 数据库连接失败: {str(e)}", style="red")
        raise


# ====================== 3. 交互式问答循环 ======================
# ====================== 3. 交互式问答循环 ======================
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

            # 显示回答标题
            console.print("\n📚 回答:", style="bold green")

            try:
                # 获取流式响应
                streaming_response = query_engine.query(query)

                # 检查是否有流式生成器
                if not hasattr(streaming_response, 'response_gen'):
                    console.print(str(streaming_response), style="green")
                    continue

                # 创建一个空的响应字符串来累积内容
                full_response = ""

                # 处理可能的生成器已执行错误
                try:
                    # 逐字输出响应
                    for token in streaming_response.response_gen:
                        if token:  # 确保token不为空
                            console.print(token, end="", style="green")
                            full_response += token
                            # 立即刷新输出缓冲区
                            console.file.flush()

                    # 添加换行符
                    console.print()

                    # 保存完整响应以便后续处理
                    streaming_response.response = full_response

                except RuntimeError as e:
                    if "generator already executing" in str(e):
                        console.print("\n⚠️ 流式响应中断，转为完整显示", style="yellow")
                        console.print(full_response, style="green")
                    else:
                        raise

                # 显示来源（如果有）
                if hasattr(streaming_response, 'source_nodes'):
                    console.print("\n🔍 来源参考:", style="bold blue")
                    for i, node in enumerate(streaming_response.source_nodes, 1):
                        console.print(f"{i}. {node.node.metadata.get('file_name', '未知文件')}")
                        console.print(f"   [相似度: {node.score:.3f}]")
                        console.print("-" * 50)

            except Exception as e:
                console.print(f"\n❌ 处理响应时出错: {str(e)}", style="red")
                continue

        except KeyboardInterrupt:
            console.print("\n👋 手动中断，再见！", style="bold red")
            break
        except Exception as e:
            console.print(f"❌ 发生错误: {str(e)}", style="red")


# ====================== 4. 主程序 ======================
if __name__ == "__main__":
    # 配置参数
    DEEPSEEK_MODEL_PATH = "/Users/yxy/work/ai/model-new/deepseek/deepseek-coder-1.3b-instruct.Q4_K_M.gguf"  # 必须使用GGUF格式
    COLLECTION_NAME = "knowledge_base_v2"

    try:
        # 初始化模型和引擎
        load_local_deepseek(DEEPSEEK_MODEL_PATH)
        query_engine = initialize_query_engine(COLLECTION_NAME)

        # 启动问答
        chat_loop(query_engine)

    except Exception as e:
        console.print(f"\n❌ 系统初始化失败: {str(e)}", style="red")


### 环境：
mac m1, python3.10

- 1，首先安装chroma
docker pull chromadb/chroma:1.0.16.dev64

```
docker run -d -p 8000:8000 \
  --name chroma-1.0.16.dev64 \
  -v /Users/yxy/work/docker/data/chroma/chroma_data.1.0.16.dev64:/chroma/chroma \
  chromadb/chroma:1.0.16.dev64
```


- 2，使用中文停用词：
> https://github.com/goto456/stopwords
>> cn_stopwords.txt


- 3，bge-small-zh-v1.5下载地址，要下载所有内容
> 下载地址 https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/main
> 
> 存放本地 /Users/yxy/work/ai/model-new/bge-small-zh-v1.5

- 4，deepseek模型下载
> 下载地址 https://huggingface.co/TheBloke/deepseek-coder-1.3b-instruct-GGUF/tree/main
> 
> 存放本地 /Users/yxy/work/ai/model-new/deepseek/

----
## 数据存储
### 基础包
```
pip install llama-index-core python-dotenv chromadb
```
### 集成和读取器
```
pip install llama-index-embeddings-huggingface llama-index-readers-file

pip install llama-index-vector-stores-chroma
```
### 文档处理依赖
```
pip install unstructured pdf2image pypdf docx2txt python-pptx
```
### 其他可能需要的依赖（根据文档类型可能需要）
```
pip install pillow pytesseract  # 图像处理相关

pip install llama-index-llms-deepseek
```


----

## 问答
### 安装依赖
```
pip install llama-cpp-python transformers chromadb llama-index rich
```

### 安装必要的包
```
pip install llama-cpp-python transformers
pip install llama-index-llms-llama-cpp  # 这是关键，安装llama_cpp集成
```


import chromadb

# 连接到 ChromaDB
client = chromadb.HttpClient(host="localhost", port=8000)  # 或 PersistentClient(path="path/to/db")

# 列出所有集合(类似索引)
collections = client.list_collections()
print("所有集合:", [col.name for col in collections])

# 获取特定集合
collection = client.get_collection("knowledge_base_v2")

# 查看集合中的记录数量
print("记录数量:", collection.count())

# 获取集合中的所有数据
results = collection.get()
print("所有数据:", results)

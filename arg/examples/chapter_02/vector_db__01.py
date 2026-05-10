# filename: weaviate_example.py

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, Property, DataType
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# 1️⃣ 连接本地 Weaviate
# ------------------------------
client = weaviate.connect_to_local(skip_init_checks=True)
print("Weaviate ready?", client.is_ready())

# ------------------------------
# 2️⃣ 创建 collection（Fruit）
# ------------------------------
collection_name = "Fruit"

# 如果 collection 已存在，先删除（可选）
existing_collections = list(client.collections.list_all(simple=True).keys())
if collection_name in existing_collections:
    client.collections.delete(collection_name)
    print(f"Deleted existing collection: {collection_name}")

# 创建 collection（无向量化器，使用自己的向量）
fruit_collection = client.collections.create(
    name=collection_name,
    vector_config=Configure.Vectors.self_provided(),
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="color", data_type=DataType.TEXT),
    ],
)
print(f"Collection '{collection_name}' created!")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Your text
texts = ["Apple is red", "Banana is yellow"]

# Convert to vectors
vectors = model.encode(texts)
fruit_collection.data.insert("name",vectors)
# ------------------------------
# 3️⃣ 插入示例数据
# ------------------------------
fruits = [
    {"name": "Apple", "color": "Red", "vector": [0.1, 0.2, 0.3]},
    {"name": "Banana", "color": "Yellow", "vector": [0.4, 0.5, 0.6]},
    {"name": "Grape", "color": "Purple", "vector": [0.7, 0.8, 0.9]},
    {"name": "Tomato", "color": "Red", "vector": [0.4, 0.2, 0.1]},
]

for fruit in fruits:
    fruit_collection.data.insert(
        properties={"name": fruit["name"], "color": fruit["color"]},
        vector=fruit["vector"],
    )
print("Sample data added!")

# ------------------------------
# 4️⃣ 执行向量查询（近邻搜索）
# ------------------------------
query_vector = [0.1, 0.2, 0.3]
result = fruit_collection.query.near_vector(
    near_vector=query_vector,
    limit=2,
    return_properties=["name", "color"],
)

print("Query result:")
for obj in result.objects:
    print(obj.properties)

# ------------------------------
# 5️⃣ 关闭连接
# ------------------------------
client.close()
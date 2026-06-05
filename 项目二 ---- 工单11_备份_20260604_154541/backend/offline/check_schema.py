"""查看 Milvus 集合 schema"""
import sys
sys.path.insert(0, ".")
from pymilvus import MilvusClient
from app.config import settings

uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
client = MilvusClient(uri=uri)
collection = settings.MILVUS_COLLECTION

# 获取 schema
schema = client.describe_collection(collection_name=collection)
print(f"集合: {collection}")
print(f"Schema: {schema}")

# 查一条数据看字段
results = client.query(
    collection_name=collection,
    filter="",
    output_fields=["*"],
    limit=1
)
if results:
    print(f"\n示例数据字段: {list(results[0].keys())}")
    for k, v in results[0].items():
        if k == "embedding":
            print(f"  {k}: [向量，长度={len(v)}]")
        else:
            print(f"  {k}: {str(v)[:100]}")

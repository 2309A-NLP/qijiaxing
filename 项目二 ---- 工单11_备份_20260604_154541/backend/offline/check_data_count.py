"""
查看 Milvus 集合实际数据量和分页查询测试
"""
import sys
sys.path.insert(0, ".")
from pymilvus import MilvusClient
from app.config import settings

uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
client = MilvusClient(uri=uri)
collection = settings.MILVUS_COLLECTION

# 获取统计信息
stats = client.get_collection_stats(collection_name=collection)
print(f"集合统计: {stats}")

# 测试不同 limit
for limit in [100, 200, 500]:
    results = client.query(
        collection_name=collection,
        filter="",
        output_fields=["chunk_id"],
        limit=limit
    )
    print(f"limit={limit}: 返回 {len(results)} 条")

# 测试用 id 范围查询
print("\n测试分页查询（使用 chunk_id 过滤）...")
all_ids = []
batch_size = 100
last_id = ""

while True:
    if last_id:
        filter_expr = f'chunk_id > "{last_id}"'
    else:
        filter_expr = ""
    
    results = client.query(
        collection_name=collection,
        filter=filter_expr,
        output_fields=["chunk_id"],
        limit=batch_size
    )
    
    if not results:
        break
    
    batch_ids = [r["chunk_id"] for r in results]
    all_ids.extend(batch_ids)
    last_id = batch_ids[-1]
    print(f"  已获取 {len(all_ids)} 条，最后ID: {last_id}")
    
    if len(results) < batch_size:
        break

print(f"\n总计: {len(all_ids)} 个文档块")

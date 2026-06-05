"""查看所有 chunk_id 的格式"""
import sys
sys.path.insert(0, ".")
from pymilvus import MilvusClient
from app.config import settings

uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
client = MilvusClient(uri=uri)
collection = settings.MILVUS_COLLECTION

# 用向量搜索获取更多数据
# 先获取一个向量
sample = client.query(
    collection_name=collection,
    filter="",
    output_fields=["embedding"],
    limit=1
)

if sample:
    vector = sample[0]["embedding"]
    
    # 用向量搜索获取 1500 条
    results = client.search(
        collection_name=collection,
        data=[vector],
        anns_field="embedding",
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=1500,
        output_fields=["chunk_id", "kb_name"]
    )
    
    if results and results[0]:
        chunk_ids = [hit["entity"]["chunk_id"] for hit in results[0]]
        kb_names = set(hit["entity"]["kb_name"] for hit in results[0])
        
        print(f"搜索返回: {len(chunk_ids)} 条")
        print(f"知识库: {kb_names}")
        
        # 统计 chunk_id 前缀
        prefixes = {}
        for cid in chunk_ids:
            prefix = cid.rsplit("_", 1)[0] if "_" in cid else "unknown"
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        print(f"\nchunk_id 前缀统计:")
        for prefix, count in sorted(prefixes.items()):
            print(f"  {prefix}: {count} 条")
        
        # 显示前 10 个和后 10 个
        print(f"\n前 10 个: {chunk_ids[:10]}")
        print(f"后 10 个: {chunk_ids[-10:]}")

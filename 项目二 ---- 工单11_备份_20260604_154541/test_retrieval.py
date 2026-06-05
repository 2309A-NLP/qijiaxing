"""Test retrieval - direct Milvus search"""
import sys
sys.path.insert(0, r"C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend")

from app.core.embedding_service import get_embedding_service
from app.db.milvus_client import MilvusClientWrapper

import app.core.embedding_service as es
es._embedding_service = None

emb = get_embedding_service()
q_vec = emb.encode("2008年中国IC市场应用结构与增长图增长率最快行业负增长")

mc = MilvusClientWrapper()
results = mc.search(embedding=q_vec, top_k=10)
print(f"Retrieved {len(results)} results")
for r in results:
    ct = r.get("content_type","?")
    sc = r.get("score",0)
    cid = r.get("chunk_id","")
    print(f"  [{ct}] score={sc:.4f} cid={cid}")

charts = [r for r in results if r.get("content_type")=="chart"]
print(f"\nChart results: {len(charts)}")
if not charts:
    print("NO CHART RESULTS RETURNED!")
else:
    print(f"Chart scores: {[round(r.get('score',0),4) for r in charts]}")

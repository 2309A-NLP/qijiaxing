"""Milvus 级检索对比：微调模型 vs 原始模型"""
import os, sys, json, random, time
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.path.join(
    os.environ.get('USERPROFILE', 'C:\\Users\\qjx'),
    '.cache', 'huggingface'
)

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = r'C:\Users\qjx\Desktop\github\项目二 ---- 工单2\data\chunks\招股说明书2_分块.jsonl'
BASE_DIR = r'C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend\offline'

# 加载分块数据
chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))
print(f"共 {len(chunks)} 个分块")

# 取前2000个chunk做候选集（覆盖全文）
candidate_chunks = [c['text'][:1000] for c in chunks[:2000]]
print(f"候选集: {len(candidate_chunks)} 个chunk")

# 测试 query（从训练数据中选有代表性的）
test_queries = [
    "公司2019年净利润是多少？",
    "发行人的主要业务是什么？",
    "公司的研发投入占营业收入的比例？",
    "前五大客户的销售金额及占比？",
    "公司的毛利率水平如何变化？",
    "募集资金的投资项目有哪些？",
    "公司的核心竞争力是什么？",
    "公司面临的主要风险因素？",
    "董事、监事、高级管理人员的薪酬情况？",
    "公司的应收账款周转率是多少？",
]

# 加载两个模型
print("\n加载模型...")
t0 = time.time()
model_orig = SentenceTransformer("BAAI/bge-base-zh-v1.5")
print(f"原始模型 加载: {time.time()-t0:.1f}s")

model_ft = SentenceTransformer(os.path.join(BASE_DIR, "finetuned-bge-base-zh-v1.5-lora"))
print(f"微调模型 加载: {time.time()-t0:.1f}s")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 批量编码候选集
print("\n编码候选集...")
t0 = time.time()
cand_orig = model_orig.encode(candidate_chunks, normalize_embeddings=True, show_progress_bar=True)
cand_ft = model_ft.encode(candidate_chunks, normalize_embeddings=True, show_progress_bar=True)
print(f"候选集编码完成: {time.time()-t0:.1f}s")

# 逐条 query 对比
print("\n" + "=" * 80)
print(f"{'Query':<40} | {'原始top1':<30} | {'微调top1':<30}")
print("=" * 80)

overlap_scores = []
for query in test_queries:
    q_prefix = QUERY_PREFIX + query
    
    emb_orig = model_orig.encode(q_prefix, normalize_embeddings=True)
    emb_ft = model_ft.encode(q_prefix, normalize_embeddings=True)
    
    sim_orig = np.dot(cand_orig, emb_orig)
    sim_ft = np.dot(cand_ft, emb_ft)
    
    top5_orig = np.argsort(-sim_orig)[:5]
    top5_ft = np.argsort(-sim_ft)[:5]
    
    # 重叠率
    overlap = len(set(top5_orig) & set(top5_ft))
    overlap_scores.append(overlap / 5)
    
    # top1的内容预览
    t1_text = candidate_chunks[top5_orig[0]][:50]
    t1_ft_text = candidate_chunks[top5_ft[0]][:50]
    
    marker = "SAME" if top5_orig[0] == top5_ft[0] else "DIFF"
    print(f"{query[:35]:<35} | {t1_text:<35} | {t1_ft_text:<35}")
    print(f"{'':35} | 分块#{top5_orig[0]} sim={sim_orig[top5_orig[0]]:.3f} | 分块#{top5_ft[0]} sim={sim_ft[top5_ft[0]]:.3f}")
    print(f"{'':35} | top5: {top5_orig[:5]} | top5: {top5_ft[:5]}")
    print(f"{'':35} | top5重叠: {overlap}/5 [{marker}]")
    print()

avg_overlap = np.mean(overlap_scores) * 100
print("=" * 80)
print(f"Top-5 平均重叠率: {avg_overlap:.0f}%")
print(f"（重叠率越高说明微调前后检索结果越一致，<50%说明微调显著改变了排序）")
print("=" * 80)

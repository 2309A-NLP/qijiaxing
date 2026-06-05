"""第二批测试：更多样化的 query"""
import os, sys, json, time
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

chunks = []
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))
candidate_chunks = [c['text'][:1000] for c in chunks]
print(f"候选集: {len(candidate_chunks)} 个chunk")

model_orig = SentenceTransformer("BAAI/bge-base-zh-v1.5")
model_ft = SentenceTransformer(os.path.join(BASE_DIR, "finetuned-bge-base-zh-v1.5-lora"))

print("编码候选集...")
t0 = time.time()
cand_orig = model_orig.encode(candidate_chunks, normalize_embeddings=True, show_progress_bar=True)
cand_ft = model_ft.encode(candidate_chunks, normalize_embeddings=True, show_progress_bar=True)
print(f"编码完成: {time.time()-t0:.1f}s")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# 第二批测试 query：更多样化
test_queries = [
    "存货周转率",                       # 短关键词
    "2019年营业收入",                   # 具体数字
    "公司有多少员工",                   # 人员信息
    "前十大股东持股比例",               # 股东信息
    "本次发行股票数量",                  # 发行信息
    "公司有没有诉讼",                   # 风险类
    "产品质量如何控制",                 # 质量控制
    "竞争对手有哪些",                   # 竞争分析
    "税收优惠政策",                     # 政策类
    "分红政策",                         # 分红
    "董监高有没有变动",                  # 人员变动
    "募投项目产能",                     # 产能
    "资产负债率",                       # 财务指标
    "研发人员数量",                     # 研发
    "销售模式",                         # 业务模式
]

print("\n" + "=" * 90)
print(f"{'Query':<30} | 原始 sim | 微调 sim | top1变化 | top5重叠")
print("=" * 90)

overlaps = []
sim_changes = []
for query in test_queries:
    q = QUERY_PREFIX + query
    emb_o = model_orig.encode(q, normalize_embeddings=True)
    emb_f = model_ft.encode(q, normalize_embeddings=True)
    
    so = np.dot(cand_orig, emb_o)
    sf = np.dot(cand_ft, emb_f)
    
    t5o = np.argsort(-so)[:5]
    t5f = np.argsort(-sf)[:5]
    
    overlap = len(set(t5o) & set(t5f))
    overlaps.append(overlap)
    
    same = "相同" if t5o[0] == t5f[0] else "不同"
    sc = "↑" if sf[t5f[0]] > so[t5o[0]] else ("↓" if sf[t5f[0]] < so[t5o[0]] else "=")
    
    print(f"{query:<30} | {so[t5o[0]]:.3f}    | {sf[t5f[0]]:.3f}    | {same} {sc} | {overlap}/5")
    # 显示top1的内容（截断）
    print(f"{'':30} | [{candidate_chunks[t5o[0]][:40]}]") 
    print(f"{'':30} | [{candidate_chunks[t5f[0]][:40]}]")
    print()

avg_overlap = np.mean(overlaps) * 20
print("=" * 90)
print(f"Top-5 平均重叠率: {avg_overlap:.0f}%  ({sum(1 for o in overlaps if o==5)}/{len(overlaps)} 条完全一致)")
print("=" * 90)

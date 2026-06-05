"""对比微调前后的 embedding 模型效果"""
import os, sys
# 强制离线模式，必须在任何 import 之前
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.path.join(
    os.environ.get('USERPROFILE', 'C:\\Users\\qjx'),
    '.cache', 'huggingface'
)

import json, time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = r'C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend\offline'

# 加载验证数据
with open(os.path.join(BASE_DIR, 'training_data', 'val.json'), 'r', encoding='utf-8') as f:
    val_data = json.load(f)

test_queries = val_data[:20]

print("=" * 60)
print("加载模型...")
print("=" * 60)

t0 = time.time()
model_orig = SentenceTransformer("BAAI/bge-base-zh-v1.5")
t1 = time.time()
print(f"原始模型 加载: {t1-t0:.1f}s")

model_ft = SentenceTransformer(os.path.join(BASE_DIR, "finetuned-bge-base-zh-v1.5-lora"))
t2 = time.time()
print(f"微调模型 加载: {t2-t1:.1f}s")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

results = []
correct_orig = 0
correct_ft = 0

print("\n" + "=" * 60)
print("逐条对比 (query vs positive vs negative)")
print("=" * 60)

for i, item in enumerate(test_queries):
    query = QUERY_PREFIX + item["query"]
    pos = item["positive"]
    neg_idx = (i + 1) % len(test_queries)
    neg = test_queries[neg_idx]["positive"]
    
    # 原始模型
    emb_q_orig = model_orig.encode(query, normalize_embeddings=True)
    emb_p_orig = model_orig.encode(pos, normalize_embeddings=True)
    emb_n_orig = model_orig.encode(neg, normalize_embeddings=True)
    sim_pos_orig = float(np.dot(emb_q_orig, emb_p_orig))
    sim_neg_orig = float(np.dot(emb_q_orig, emb_n_orig))
    correct_orig += 1 if sim_pos_orig > sim_neg_orig else 0
    
    # 微调模型
    emb_q_ft = model_ft.encode(query, normalize_embeddings=True)
    emb_p_ft = model_ft.encode(pos, normalize_embeddings=True)
    emb_n_ft = model_ft.encode(neg, normalize_embeddings=True)
    sim_pos_ft = float(np.dot(emb_q_ft, emb_p_ft))
    sim_neg_ft = float(np.dot(emb_q_ft, emb_n_ft))
    correct_ft += 1 if sim_pos_ft > sim_neg_ft else 0
    
    sep_orig = sim_pos_orig - sim_neg_orig
    sep_ft = sim_pos_ft - sim_neg_ft
    
    results.append({
        'query': item['query'][:50],
        'orig_pos': sim_pos_orig,
        'orig_neg': sim_neg_orig,
        'orig_correct': sim_pos_orig > sim_neg_orig,
        'ft_pos': sim_pos_ft,
        'ft_neg': sim_neg_ft,
        'ft_correct': sim_pos_ft > sim_neg_ft,
        'sep_orig': sep_orig,
        'sep_ft': sep_ft,
    })
    
    status_orig = "O" if sim_pos_orig > sim_neg_orig else "X"
    status_ft = "O" if sim_pos_ft > sim_neg_ft else "X"
    print(f"[{i+1:2d}] {status_orig}原始: pos={sim_pos_orig:.3f} neg={sim_neg_orig:.3f} sep={sep_orig:+.3f} | "
          f"{status_ft}微调: pos={sim_pos_ft:.3f} neg={sim_neg_ft:.3f} sep={sep_ft:+.3f}")

print("\n" + "=" * 60)
print("汇总")
print("=" * 60)
print(f"原始模型: {correct_orig}/{len(test_queries)} 正确 ({100*correct_orig/len(test_queries):.0f}%)")
print(f"微调模型: {correct_ft}/{len(test_queries)} 正确 ({100*correct_ft/len(test_queries):.0f}%)")
print(f"微调改善: {correct_ft - correct_orig:+d} 条")

avg_sep_orig = np.mean([r['sep_orig'] for r in results])
avg_sep_ft = np.mean([r['sep_ft'] for r in results])
print(f"正负例分隔度 原始: {avg_sep_orig:.3f} | 微调: {avg_sep_ft:.3f} "
      f"({'改善' if avg_sep_ft > avg_sep_orig else '下降'}: {abs(avg_sep_ft-avg_sep_orig):.3f})")

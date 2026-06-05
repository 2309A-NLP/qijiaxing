"""
run_pipeline.py — 一键运行完整离线管道
用法: python run_pipeline.py
流程: MinerU解析 -> 文本分块 -> 向量化 -> 导入Milvus
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
KB_NAME = "招股说明书2"

os.chdir(str(PROJECT_DIR))
os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KB_NAME"] = KB_NAME

PYTHON = sys.executable

steps = [
    ("[1/4] MinerU 解析 PDF", ["offline/parse_pdf_mineru.py", KB_NAME]),
    ("[2/4] 文本分块", ["offline/chunk_text.py"]),
    ("[3/4] 向量化 Embedding", ["offline/generate_embeddings.py", KB_NAME]),
    ("[4/4] 导入 Milvus", ["offline/import_to_milvus.py"]),
]

for title, args in steps:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    result = subprocess.run([PYTHON] + args)
    if result.returncode != 0:
        print(f"[失败] {title}，退出码 {result.returncode}")
        sys.exit(result.returncode)
    print(f"[完成] {title}\n")

print("=" * 60)
print("  全流程执行完成！")
print("=" * 60)

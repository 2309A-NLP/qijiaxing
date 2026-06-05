"""
config.py — 离线管道配置文件
============================
作用：集中管理离线管道的所有配置参数
原理：使用普通常量（离线批处理不需要动态配置）
"""

import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# 路径配置
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"

# 知识库配置（v2.0: 支持多知识库）
KB_CONFIGS = {
    "招股说明书1": {
        "pdf_path": DATA_DIR / "招股说明书1.pdf",
        "parsed_json_path": DATA_DIR / "parsed" / "招股说明书1_pages.json",
    },
    "招股说明书2": {
        "pdf_path": DATA_DIR / "招股说明书2.pdf",
        "parsed_json_path": DATA_DIR / "parsed" / "招股说明书2_pages.json",
    },
}

def get_kb_paths(kb_name: str) -> dict:
    """获取指定知识库的文件路径"""
    if kb_name not in KB_CONFIGS:
        raise ValueError(f"未知知识库: {kb_name}，可选: {list(KB_CONFIGS.keys())}")
    return KB_CONFIGS[kb_name]

PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_DIR = DATA_DIR / "chunks"

# 兼容旧版脚本的路径常量
PARSED_JSON_PATH = PARSED_DIR / "招股说明书_原始文本.json"
CHUNKS_JSONL_PATH = CHUNKS_DIR / "招股说明书_分块.jsonl"
EMBEDDINGS_JSONL_PATH = CHUNKS_DIR / "招股说明书_带向量.jsonl"

# 知识库列表（供import_to_milvus遍历）
KNOWLEDGE_BASES = KB_CONFIGS

# =============================================================================
# 分块配置
# =============================================================================
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 150
MIN_LINE_LENGTH = 30

# =============================================================================
# 向量化配置
# =============================================================================
EMBEDDING_MODEL_PATH = "C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-base-zh-v1.5"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 768  # bge-base-zh-v1.5

# =============================================================================
# Milvus配置
# =============================================================================
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "prospectus"
MILVUS_BATCH_SIZE = 100
REBUILD_COLLECTION = False

# =============================================================================
# 日志配置
# =============================================================================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(name: str) -> logging.Logger:
    """创建带控制台和文件输出的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    if logger.handlers:
        return logger
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{name}_{RUN_TIMESTAMP}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    return logger

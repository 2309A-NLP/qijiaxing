"""
config.py — 项目配置文件
=========================
作用：集中管理整个项目的配置参数，包括路径、常量、日志设置
原理：所有脚本都导入此文件，确保配置统一，避免硬编码分散在各处
"""

import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# 项目根目录
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# 日志配置
# =============================================================================
LOG_DIR = PROJECT_DIR / "logs"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================================================================
# 数据目录和文件路径
# =============================================================================
DATA_DIR = PROJECT_DIR / "data"
PDF_PATH = DATA_DIR / "招股说明书1.pdf"
PARSED_DIR = DATA_DIR / "parsed"
PARSED_JSON_PATH = PARSED_DIR / "招股说明书_原始文本.json"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_JSONL_PATH = CHUNKS_DIR / "招股说明书_分块.jsonl"
EMBEDDINGS_JSONL_PATH = CHUNKS_DIR / "招股说明书_带向量.jsonl"

# =============================================================================
# 知识库定义（多KB支持）
# =============================================================================
KNOWLEDGE_BASES = {
    "招股说明书1": {
        "parser": "mineru",
        "pdf_path": DATA_DIR / "招股说明书1.pdf",
    },
    "招股说明书2": {
        "parser": "mineru",
        "pdf_path": DATA_DIR / "招股说明书2.pdf",
    },
}


def get_kb_paths(kb_name: str) -> dict:
    """获取知识库的所有路径"""
    return {
        "pdf_path": DATA_DIR / f"{kb_name}.pdf",
        "parsed_dir": PARSED_DIR / kb_name,
        "parsed_json_path": PARSED_DIR / f"{kb_name}_原始文本.json",
        "chunks_jsonl_path": CHUNKS_DIR / f"{kb_name}_分块.jsonl",
        "embeddings_jsonl_path": CHUNKS_DIR / f"{kb_name}_带向量.jsonl",
    }


# =============================================================================
# 文本分块参数
# =============================================================================
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 150
MIN_LINE_LENGTH = 30

# =============================================================================
# PDF解析参数
# =============================================================================
PDF_HEADER_KEYWORDS = ["武汉兴图新科", "招股意向书", "招股说明书"]
PDF_PAGE_NUMBER_PATTERN = r"^\d+-\d+-\d+$"

# =============================================================================
# Embedding模型配置
# =============================================================================
EMBEDDING_MODEL_PATH = "C:/Users/qjx/.cache/modelscope/hub/models/BAAI/bge-m3"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIM = 1024

# =============================================================================
# Milvus向量数据库配置
# =============================================================================
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "prospectus"
MILVUS_BATCH_SIZE = 100
REBUILD_COLLECTION = False  # 已手动删除集合，这里设False即可

# =============================================================================
# 质量检查阈值
# =============================================================================
QUALITY_MIN_CHUNKS = 100
QUALITY_MAX_EMPTY_RATIO = 0.05
QUALITY_MIN_EMBEDDING_RATIO = 0.95


# =============================================================================
# 日志设置函数
# =============================================================================
def setup_logger(name: str) -> logging.Logger:
    """创建并配置一个日志记录器"""
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

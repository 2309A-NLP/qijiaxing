# -*- coding: utf-8 -*-
"""检查所有模块能否正常导入

作用：验证后台所有核心模块的导入是否正常，定位导入错误。
原理：使用try/except逐个模块导入，成功打"✅"标记，失败记录错误信息。
用途：git pull或环境变更后快速检查依赖是否完整。

面试官可能问：
  Q: 为什么要写check_imports.py，pip install不就能检查依赖吗？
  A: pip install只检查requirements.txt里的包是否安装，不检查
     项目内部的import路径是否正确。比如sys.path.insert写错了路径、
     某个模块内部有语法错误、相对导入路径不对——这些pip检查不到。
     check_imports逐个导入模块，能精准定位"哪个文件导入失败、原因是什么"。

  Q: 为什么不写成单元测试？
  A: 可以写成unittest，但check_imports.py是给运维人员手动跑的脚本
     （git pull后跑一下确认没坏），不需要测试框架。
     如果要自动化，可以集成到CI流程中。
"""
import sys
import os

# 添加backend目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []

# 检查Query理解模块
try:
    from app.core.query_understanding import QueryUnderstandingService, IntentType, Entity, QueryUnderstandingResult
    print("✅ query_understanding.py - OK")
except Exception as e:
    errors.append(f"❌ query_understanding.py - {e}")

# 检查BM25模块
try:
    from app.core.bm25_retriever import BM25Retriever, HybridRetriever
    print("✅ bm25_retriever.py - OK")
except Exception as e:
    errors.append(f"❌ bm25_retriever.py - {e}")

# 检查重排序模块
try:
    from app.core.reranker import CrossEncoderReranker, RerankerService, RerankResult
    print("✅ reranker.py - OK")
except Exception as e:
    errors.append(f"❌ reranker.py - {e}")

# 检查Redis模块
try:
    from app.db.redis_client import RedisCache, SessionManager, QueryCache, CacheManager
    print("✅ redis_client.py - OK")
except Exception as e:
    errors.append(f"❌ redis_client.py - {e}")

# 检查其他核心模块
try:
    from app.core.embedding_service import EmbeddingService
    print("✅ embedding_service.py - OK")
except Exception as e:
    errors.append(f"❌ embedding_service.py - {e}")

try:
    from app.core.llm_client import LLMClient
    print("✅ llm_client.py - OK")
except Exception as e:
    errors.append(f"❌ llm_client.py - {e}")

try:
    from app.core.llm_router import LLMRouter
    print("✅ llm_router.py - OK")
except Exception as e:
    errors.append(f"❌ llm_router.py - {e}")

try:
    from app.core.retrieval import RetrievalService
    print("✅ retrieval.py - OK")
except Exception as e:
    errors.append(f"❌ retrieval.py - {e}")

try:
    from app.db.milvus_client import MilvusClient
    print("✅ milvus_client.py - OK")
except Exception as e:
    errors.append(f"❌ milvus_client.py - {e}")

# 输出结果
print("\n" + "="*50)
if errors:
    print("检查失败:")
    for err in errors:
        print(f"  {err}")
else:
    print("所有模块检查通过！")

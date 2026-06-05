"""
backend/app/core/llm_router.py — LLM智能调度器（异步版）
================================================================
作用：根据任务复杂度智能选择LLM推理引擎，实现 async 调用链
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional

import httpx

from ..config import settings
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    作用：LLM智能调度器，自动选择最优引擎，async 版本
    """

    COMPLEX_KEYWORDS = [
        "分析", "比较", "对比", "评估", "评价",
        "趋势", "变化", "发展", "预测", "展望",
        "总结", "归纳", "概述", "概括", "提炼",
        "为什么", "原因", "影响", "意义", "作用",
        "代码", "程序", "脚本", "函数", "算法",
        "详细", "深入", "全面", "系统", "综合",
    ]

    SIMPLE_PATTERNS = [
        r"^\d+年.*是多少",
        r"^什么是",
        r"^.*的?定义",
        r"^第?\d+页",
        r"^.*是\d+",
    ]

    def __init__(self):
        self.complex_threshold = settings.LLM_COMPLEX_THRESHOLD

        # 检测服务（用同步 httpx，因为 __init__ 不能 async）
        sglang_check_host = "http://172.23.190.86:30000"
        self.sglang_available = self._check_service(sglang_check_host, "SGLang")
        self.ollama_available = self._check_service(settings.OLLAMA_HOST, "Ollama")

        self.sglang_client = None
        self.ollama_client = None

        if self.sglang_available:
            self.sglang_client = self._create_client(
                provider="sglang",
                host=settings.SGLANG_HOST,
                model=settings.SGLANG_MODEL
            )
            logger.info(f"✅ SGLang 已连接: {settings.SGLANG_MODEL}")

        if self.ollama_available:
            self.ollama_client = self._create_client(
                provider="ollama",
                host=settings.OLLAMA_HOST,
                model=settings.OLLAMA_MODEL
            )
            logger.info(f"✅ Ollama 已连接: {settings.OLLAMA_MODEL}")

        if not self.sglang_available and not self.ollama_available:
            logger.error("❌ 没有可用的LLM服务！")
            logger.error("请启动 SGLang (端口30000) 或 Ollama (端口11434)")
            raise RuntimeError("没有可用的LLM服务")

        logger.info("LLM智能调度器初始化完成")

    def _check_service(self, host: str, name: str) -> bool:
        """同步检测服务（__init__中用，不改）"""
        try:
            root_url = host.replace("/v1/chat/completions", "").replace("/api/chat", "")
            response = httpx.get(root_url, timeout=2)
            logger.debug(f"{name} 状态码: {response.status_code}")
            return True
        except httpx.ConnectError:
            logger.debug(f"{name} 未启动 ({host})")
            return False
        except Exception as e:
            logger.debug(f"{name} 检测失败: {e}")
            return False

    def _create_client(self, provider: str, host: str, model: str) -> LLMClient:
        """创建 LLMClient（同步，仅初始化属性）"""
        client = LLMClient.__new__(LLMClient)
        client.provider = provider
        if provider == "sglang":
            client.host = "http://172.23.190.86:30000"
            client.model = "default"
        else:
            client.host = host
            client.model = model
        client.api_key = ""
        client.timeout = settings.LLM_TIMEOUT
        client.max_retries = settings.LLM_MAX_RETRIES
        if provider == "sglang":
            client.chat_url = f"{client.host}/v1/chat/completions"
        else:
            client.chat_url = f"{client.host}/api/chat"
        return client

    async def generate(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """async 智能生成——选引擎 → await 生成 → 失败切备用"""
        client = self._select_client(query, history)

        try:
            return await client.generate(query, context, history)
        except Exception as e:
            logger.warning(f"主引擎失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"切换到备用引擎: {fallback.provider}")
                return await fallback.generate(query, context, history)
            raise

    async def generate_stream(
        self,
        query: str,
        context: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ):
        """async 流式智能生成——选引擎 → async for 逐 token"""
        client = self._select_client(query, history)

        try:
            async for token in client.generate_stream(query, context, history):
                yield token
        except Exception as e:
            logger.warning(f"主引擎失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"切换到备用引擎: {fallback.provider}")
                async for token in fallback.generate_stream(query, context, history):
                    yield token
            raise

    def _select_client(self, query: str, history: Optional[List[Dict]] = None) -> LLMClient:
        if self.sglang_available and not self.ollama_available:
            return self.sglang_client
        if self.ollama_available and not self.sglang_available:
            return self.ollama_client
        complexity = self._analyze_complexity(query, history)
        if complexity == "complex":
            logger.info(f"[智能调度] 复杂 → SGLang | {query[:50]}...")
            return self.sglang_client
        else:
            logger.info(f"[智能调度] 简单 → Ollama | {query[:50]}...")
            return self.ollama_client

    def _get_fallback_client(self, current_client: LLMClient) -> Optional[LLMClient]:
        if current_client == self.sglang_client and self.ollama_available:
            return self.ollama_client
        if current_client == self.ollama_client and self.sglang_available:
            return self.sglang_client
        return None

    def _analyze_complexity(self, query: str, history: Optional[List[Dict]] = None) -> str:
        if history and len(history) > 0:
            return "complex"
        for keyword in self.COMPLEX_KEYWORDS:
            if keyword in query:
                return "complex"
        if len(query) > self.complex_threshold:
            return "complex"
        for pattern in self.SIMPLE_PATTERNS:
            if re.match(pattern, query):
                return "simple"
        return "simple"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sglang": {"available": self.sglang_available, "host": settings.SGLANG_HOST if self.sglang_available else None, "model": settings.SGLANG_MODEL if self.sglang_available else None},
            "ollama": {"available": self.ollama_available, "host": settings.OLLAMA_HOST if self.ollama_available else None, "model": settings.OLLAMA_MODEL if self.ollama_available else None},
            "threshold": self.complex_threshold,
        }


_llm_router = None


def get_llm_router() -> LLMRouter:
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router

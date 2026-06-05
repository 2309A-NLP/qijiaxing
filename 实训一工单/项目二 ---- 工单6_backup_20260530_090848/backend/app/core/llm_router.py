"""backend/app/core/llm_router.py — LLM智能路由（Harness抽象层版）

借鉴 HiveWard 的 Harness 设计模式：
  每个 LLM provider 是一个独立的 Harness 配置，
  Router 只负责选择和故障转移，不关心具体实现细节。
  新增 provider 只需在 HARNESS_CONFIGS 中添加一条配置，零代码改动。

优先级：问候语 → 低价引擎 | 复杂问题 → 按 priority 排序
故障转移：当前引擎失败 → 按 priority 顺序尝试下一个
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from ..config import settings
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# 问候语模式（匹配这些直接走低价引擎，不需要大模型）
GREETING_PATTERNS = [
    r"^(你好|您好|嗨|哈喽|hi|hello|hey|早上好|下午好|晚上好|早安|晚安)[\s!！。.~～]*$",
]


@dataclass
class LLMHarness:
    """LLM Harness —— 单个 provider 的完整配置。

    借鉴 HiveWard 的 Harness 抽象：每个 CLI 工具（Claude Code / Cursor / OpenCode）
    对应一个 Harness，统一管理权限、配置、状态。
    这里把同样的思路用在 LLM provider 上：每个 provider（SGLang / DeepSeek / Ollama）
    对应一个 Harness，统一管理连接、模型、优先级、健康检查。

    设计决策（面试话术）：
      "为什么不继续用 self.sglang_client / self.deepseek_client 的硬编码方式？
       因为每加一个 provider 要改 4 个地方（__init__、_create_client、
       _select_client、_get_fallback_client），这是典型的霰弹式修改。
       Harness 把每个 provider 封装为独立单元，Router 只通过统一接口操作，
       新增 provider 只需加一条配置。"
    """
    name: str                          # provider 名称（sglang/deepseek/ollama/openai...）
    host: str                          # API 地址
    model: str                         # 模型名
    api_key: str = ""                  # API Key（本地模型为空）
    priority: int = 0                  # 越大越优先（问候语场景用 priority=0 的低价引擎）
    is_greeting_engine: bool = False   # 是否作为问候语专用引擎
    health_path: str = "/health"       # 健康检查路径
    chat_path: str = ""                # 聊天补全路径（自动推导）
    cloud_api: bool = False            # 是否云端API（有Key就认为可用）
    client: Optional[LLMClient] = None # 运行时创建的客户端
    available: bool = False            # 运行时健康状态

    def derive_chat_path(self) -> str:
        """根据 provider 类型推导聊天 API 路径。"""
        if self.chat_path:
            return self.chat_path
        if self.name == "ollama":
            return f"{self.host}/api/chat"
        # OpenAI 兼容协议（sglang/deepseek/openai/...）
        return f"{self.host}/v1/chat/completions"


# ──────────────────────────────────────────────
# Harness 配置注册表（新增 provider 只改这里）
# ──────────────────────────────────────────────
HARNESS_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "sglang",
        "host_setting": "SGLANG_HOST",
        "model_setting": "SGLANG_MODEL",
        "key_setting": None,
        "priority": 100,
        "is_greeting_engine": False,
        "cloud_api": False,
    },
    {
        "name": "deepseek",
        "host_setting": "DEEPSEEK_HOST",
        "model_setting": "DEEPSEEK_MODEL",
        "key_setting": "DEEPSEEK_KEY",
        "priority": 50,
        "is_greeting_engine": False,
        "cloud_api": True,
    },
    {
        "name": "ollama",
        "host_setting": "OLLAMA_HOST",
        "model_setting": "OLLAMA_MODEL",
        "key_setting": None,
        "priority": 10,
        "is_greeting_engine": True,   # 问候语默认走 Ollama（省钱）
        "cloud_api": False,
    },
]


class LLMRouter:
    """LLM路由器：基于 Harness 抽象层统一管理多个 LLM 引擎。

    核心设计：
      1. Harness 注册表驱动 —— 所有 provider 配置集中在 HARNESS_CONFIGS
      2. 自动发现 —— 启动时遍历配置，自动健康检查和创建客户端
      3. 优先级排序 —— 按 priority 选择引擎，不再硬编码 if/elif 链
      4. 故障转移 —— 当前引擎失败，自动尝试下一个
    """

    def __init__(self):
        self.harnesses: List[LLMHarness] = []
        self._discover_harnesses()

        if not self.harnesses:
            logger.error("❌ 没有可用的LLM服务！请启动 SGLang (端口30000) 或 Ollama (端口11434) 或配置 DeepSeek API")

    def _discover_harnesses(self):
        """遍历配置注册表，自动发现可用的 LLM Harness。"""
        for cfg in HARNESS_CONFIGS:
            host = getattr(settings, cfg["host_setting"], "")
            model = getattr(settings, cfg["model_setting"], "")
            api_key = getattr(settings, cfg["key_setting"], "") if cfg["key_setting"] else ""

            if not host or not model:
                continue

            harness = LLMHarness(
                name=cfg["name"],
                host=host,
                model=model,
                api_key=api_key,
                priority=cfg["priority"],
                is_greeting_engine=cfg["is_greeting_engine"],
                cloud_api=cfg["cloud_api"],
            )

            # 健康检查
            harness.available = self._check_health(harness)
            if harness.available:
                harness.client = self._create_client(harness)
                logger.info(f"✅ {harness.name} 已连接: {model} (priority={harness.priority})")
            else:
                logger.info(f"⏭️ {harness.name} 不可用，跳过")

            self.harnesses.append(harness)

    def _check_health(self, harness: LLMHarness) -> bool:
        """检查 Harness 对应的 LLM 服务是否健康。"""
        try:
            import httpx
            if harness.cloud_api:
                return bool(harness.api_key)
            root_url = harness.host.replace("/v1/chat/completions", "").replace("/api/chat", "")
            resp = httpx.get(f"{root_url}{harness.health_path}", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def _create_client(self, harness: LLMHarness) -> LLMClient:
        """为 Harness 创建 LLMClient（绕过 __init__ 的单一 provider 限制）。"""
        client = LLMClient.__new__(LLMClient)
        client.provider = harness.name
        client.host = harness.host
        client.model = harness.model
        client.api_key = harness.api_key
        client.max_retries = 3
        client.timeout = 60
        client.chat_url = harness.derive_chat_path()
        return client

    def _get_available(self, exclude: Optional[LLMClient] = None) -> List[LLMHarness]:
        """获取所有可用 Harness，按 priority 降序排列。"""
        return sorted(
            [h for h in self.harnesses if h.available and h.client and h.client != exclude],
            key=lambda h: h.priority,
            reverse=True,
        )

    def _is_greeting(self, query: str) -> bool:
        return any(re.match(p, query.strip(), re.IGNORECASE) for p in GREETING_PATTERNS)

    def _select_client(self, query: str, history: Optional[List[Dict]] = None) -> LLMClient:
        """智能选择 LLM 客户端（基于 Harness 优先级）。

        规则：
          1. 问候语 → 优先选 is_greeting_engine=True 的 Harness
          2. 其他 → 按 priority 降序选第一个可用的
        """
        if self._is_greeting(query):
            greeting_harness = next(
                (h for h in self.harnesses if h.is_greeting_engine and h.available and h.client),
                None,
            )
            if greeting_harness:
                logger.info(f"[智能调度] 问候语 → {greeting_harness.name} | {query[:50]}")
                return greeting_harness.client  # type: ignore

        available = self._get_available()
        if not available:
            raise RuntimeError("没有可用的LLM服务")

        selected = available[0]
        logger.info(f"[智能调度] {selected.name} (priority={selected.priority}) | {query[:50]}...")
        return selected.client  # type: ignore

    def _get_fallback_client(self, current_client: LLMClient) -> Optional[LLMClient]:
        """获取备用客户端（故障转移，按 priority 降序）。"""
        available = self._get_available(exclude=current_client)
        return available[0].client if available else None

    async def generate(self, query: str, context: List[str], history: Optional[List[Dict]] = None) -> str:
        """生成答案（非流式）"""
        client = self._select_client(query, history)
        try:
            return await client.generate(query=query, context=context, history=history)
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"故障转移到备用引擎: {fallback.provider}")
                return await fallback.generate(query=query, context=context, history=history)
            raise

    async def generate_stream(self, query: str, context: List[str], history: Optional[List[Dict]] = None):
        """生成答案（流式）"""
        client = self._select_client(query, history)
        try:
            async for chunk in client.generate_stream(query=query, context=context, history=history):
                yield chunk
        except Exception as e:
            logger.warning(f"LLM流式调用失败: {e}")
            fallback = self._get_fallback_client(client)
            if fallback:
                logger.info(f"故障转移到备用引擎: {fallback.provider}")
                async for chunk in fallback.generate_stream(query=query, context=context, history=history):
                    yield chunk
            else:
                raise


_llm_router = None


def get_llm_router() -> LLMRouter:
    """单例工厂"""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router

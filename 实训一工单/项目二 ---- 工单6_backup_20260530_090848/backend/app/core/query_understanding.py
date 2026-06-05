# -*- coding: utf-8 -*-
"""
Query理解模块（LLM驱动版）
功能：意图识别、实体提取、问题分解、消歧处理 + 公司实体识别

v3.0 改进：
  - 意图识别：正则 → LLM直接判断（支持中英文、同义词）
  - 实体提取：jieba → LLM结构化输出（更准确）
  - 保持接口不变，调用方无需修改
"""

import json
import logging
import re
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .kb_mapping import resolve_kb_name, list_knowledge_bases

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """
    作用：意图类型枚举

    定义系统能识别的用户意图类型：
    - FINANCIAL: 财务数据类问题（如：收入、利润、资产等）
    - RISK: 风险因素类问题（如：经营风险、市场风险等）
    - BUSINESS: 业务介绍类问题（如：主营业务、产品服务等）
    - LEGAL: 法律合规类问题（如：诉讼、处罚等）
    - MANAGEMENT: 管理层信息类问题（如：董事长、总经理等）
    - COMPARISON: 对比分析类问题（如：对比两家公司的财务数据）
    - GREETING: 问候/闲聊类（如：你好、谢谢等）
    - UNCLEAR: 意图不明确，无法分类
    - OUT_OF_SCOPE: 超出系统知识范围
    """
    FINANCIAL = "financial"
    RISK = "risk"
    BUSINESS = "business"
    LEGAL = "legal"
    MANAGEMENT = "management"
    COMPARISON = "comparison"
    GREETING = "greeting"
    UNCLEAR = "unclear"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class Entity:
    """
    作用：实体数据类

    用于存储从用户问题中提取的实体信息
    """
    text: str
    type: str
    start: int
    end: int


@dataclass
class QueryUnderstandingResult:
    """
    作用：Query理解结果数据类

    存储完整的Query理解结果，供后续模块使用
    """
    original_query: str
    intent: IntentType
    intent_confidence: float
    entities: List[Entity]
    normalized_query: str
    sub_queries: List[str]
    needs_clarification: bool
    clarification_prompt: Optional[str]
    detected_kb_name: Optional[str] = None


# LLM理解Prompt模板
UNDERSTAND_PROMPT = """你是一个Query理解引擎。分析用户问题，输出JSON。

用户问题：{query}

可选意图类型：
- FINANCIAL: 财务相关（收入、利润、资产、负债、现金流、revenue、profit、assets等）
- RISK: 风险相关（风险、挑战、问题、不确定性、risk、challenge等）
- BUSINESS: 业务相关（产品、服务、市场、客户、商业模式、business、product等）
- LEGAL: 法律合规（诉讼、处罚、合规、lawsuit、compliance等）
- MANAGEMENT: 管理层（董事长、高管、股东、management、executive等）
- COMPARISON: 对比分析（对比、比较、vs、compare等）
- GREETING: 问候/闲聊（你好、谢谢、hello、thanks等）
- OUT_OF_SCOPE: 超出招股书知识范围（天气、娱乐等无关问题）
- UNCLEAR: 无法判断意图

输出JSON格式（不要输出其他内容）：
{{
  "intent": "意图类型",
  "confidence": 0.95,
  "entities": [
    {{"type": "company|person|time|metric|money", "value": "提取的值"}}
  ],
  "normalized_query": "归一化后的问题（去年→2024年，口语→书面语）",
  "sub_queries": ["拆解的子问题，无则为空数组"],
  "needs_clarification": false,
  "clarification_prompt": null
}}"""

# 指代消解Prompt
ANAPHORA_PROMPT = """你是一个指代消解引擎。根据对话历史，将当前问题中的指代词替换为具体实体。

对话历史：
{history}

当前问题：{query}

规则：
- "他们"、"该公司"、"其"、"这家"等指代词 → 替换为历史中提到的公司名
- "去年"、"前年"等时间指代 → 替换为具体年份
- 如果没有指代词或无法消解，返回原问题

只返回消解后的问题文本，不要输出其他内容。"""


class QueryUnderstandingService:
    """
    作用：Query理解服务类（LLM驱动版）

    v3.0 改进：
      - 意图识别+实体提取+归一化：一个LLM调用完成
      - 天然支持中英文，无需维护多套规则
      - 新领域只需改prompt，不用改代码
    """

    def __init__(self, llm_router=None):
        """
        作用：初始化Query理解服务

        Args:
            llm_router: LLM路由器实例，用于调用大模型
        """
        self.llm_router = llm_router

    def understand(self, query: str, conversation_history: Optional[List[Dict]] = None) -> QueryUnderstandingResult:
        """
        作用：理解用户Query的主入口方法

        执行完整的Query理解流程：
        1. 公司实体识别（基于知识库映射）
        2. 指代消解（如有对话历史）
        3. LLM理解（意图+实体+归一化+分解）
        4. kb_name消歧

        Args:
            query: 用户原始问题
            conversation_history: 对话历史，用于指代消解

        Returns:
            QueryUnderstandingResult: 包含完整理解结果的数据类
        """
        original_query = query

        # 步骤1：公司实体识别（基于知识库映射，不依赖LLM）
        detected_kb_name = resolve_kb_name(query)

        # 步骤2：指代消解（如有对话历史）
        if conversation_history and len(conversation_history) > 0:
            resolved = self._resolve_anaphora(query, conversation_history)
            if resolved != query:
                logger.info(f"💬 指代消解: '{query}' → '{resolved}'")
                query = resolved
                detected_kb_name = resolve_kb_name(query)

        # 步骤3：LLM理解（意图+实体+归一化+分解）
        llm_result = self._llm_understand(query)

        # 步骤4：解析LLM结果
        intent = self._parse_intent(llm_result.get("intent", "UNCLEAR"))
        confidence = float(llm_result.get("confidence", 0.5))
        entities = self._parse_entities(llm_result.get("entities", []), query)
        normalized_query = llm_result.get("normalized_query", query)
        sub_queries = llm_result.get("sub_queries", [])
        needs_clarification = llm_result.get("needs_clarification", False)
        clarification_prompt = llm_result.get("clarification_prompt")

        # 步骤5：kb_name消歧（如果LLM没判断需要澄清，但公司名模糊）
        if not needs_clarification and detected_kb_name is None:
            kb_needs = self._check_kb_clarification(query, intent, conversation_history)
            if kb_needs:
                needs_clarification, clarification_prompt = kb_needs

        return QueryUnderstandingResult(
            original_query=original_query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            normalized_query=normalized_query,
            sub_queries=sub_queries,
            needs_clarification=needs_clarification,
            clarification_prompt=clarification_prompt,
            detected_kb_name=detected_kb_name,
        )

    def _llm_understand(self, query: str) -> Dict:
        """
        作用：调用LLM理解Query

        Args:
            query: 用户问题

        Returns:
            Dict: LLM返回的JSON结果
        """
        if not self.llm_router:
            logger.warning("LLM Router未初始化，返回默认结果")
            return {
                "intent": "UNCLEAR",
                "confidence": 0.3,
                "entities": [],
                "normalized_query": query,
                "sub_queries": [],
                "needs_clarification": False,
                "clarification_prompt": None
            }

        prompt = UNDERSTAND_PROMPT.format(query=query)

        try:
            # 使用同步方式调用异步LLM
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # 如果已有事件循环（如在FastAPI中），用线程池
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.llm_router.generate(query=prompt, context=[]))
                    response = future.result(timeout=15)
            else:
                response = asyncio.run(self.llm_router.generate(query=prompt, context=[]))

            # 解析JSON
            # 提取JSON部分（LLM可能输出额外文本）
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning(f"LLM返回无法解析为JSON: {response[:200]}")
                return {"intent": "UNCLEAR", "confidence": 0.3}

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return {"intent": "UNCLEAR", "confidence": 0.3}

    def _resolve_anaphora(self, query: str, history: List[Dict]) -> str:
        """
        作用：指代消解

        Args:
            query: 当前问题
            history: 对话历史

        Returns:
            str: 消解后的问题
        """
        if not self.llm_router or not history:
            return query

        # 构造历史文本
        history_text = "\n".join(
            f"{h.get('role', 'user')}: {h.get('content', '')}"
            for h in history[-5:]  # 只取最近5条
        )

        prompt = ANAPHORA_PROMPT.format(history=history_text, query=query)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.llm_router.generate(query=prompt, context=[]))
                    response = future.result(timeout=15)
            else:
                response = asyncio.run(self.llm_router.generate(query=prompt, context=[]))

            return response.strip() if response.strip() else query

        except Exception as e:
            logger.error(f"指代消解失败: {e}")
            return query

    def _parse_intent(self, intent_str: str) -> IntentType:
        """
        作用：解析意图字符串为枚举

        Args:
            intent_str: 意图字符串

        Returns:
            IntentType: 意图枚举
        """
        intent_map = {
            "FINANCIAL": IntentType.FINANCIAL,
            "RISK": IntentType.RISK,
            "BUSINESS": IntentType.BUSINESS,
            "LEGAL": IntentType.LEGAL,
            "MANAGEMENT": IntentType.MANAGEMENT,
            "COMPARISON": IntentType.COMPARISON,
            "GREETING": IntentType.GREETING,
            "OUT_OF_SCOPE": IntentType.OUT_OF_SCOPE,
            "UNCLEAR": IntentType.UNCLEAR,
        }
        return intent_map.get(intent_str.upper(), IntentType.UNCLEAR)

    def _parse_entities(self, entities_list: List[Dict], query: str) -> List[Entity]:
        """
        作用：解析实体列表

        Args:
            entities_list: LLM返回的实体列表
            query: 原始查询

        Returns:
            List[Entity]: 实体列表
        """
        entities = []
        for ent in entities_list:
            value = ent.get("value", "")
            if value and value in query:
                start = query.index(value)
                entities.append(Entity(
                    text=value,
                    type=ent.get("type", "unknown"),
                    start=start,
                    end=start + len(value)
                ))
        return entities

    def _check_kb_clarification(
        self,
        query: str,
        intent: IntentType,
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Tuple[bool, str]]:
        """
        作用：检查是否因公司名模糊需要向用户澄清

        Args:
            query: 用户问题
            intent: 识别的意图
            conversation_history: 对话历史

        Returns:
            Optional[Tuple[bool, str]]: (True, 澄清提示语) 或 None
        """
        if intent in (IntentType.COMPARISON, IntentType.GREETING, IntentType.UNCLEAR):
            return None

        # 检查问题是否包含公司相关指标
        company_indicators = [
            r"(公司|企业|发行人|本[公企])",
            r"(营收|收入|利润|资产|负债|风险|业务|产品|技术|专利|股本|发行)",
            r"(company|revenue|profit|business|risk)",
        ]
        has_indicator = any(re.search(p, query, re.IGNORECASE) for p in company_indicators)
        if not has_indicator:
            return None

        # 检查对话历史中是否有公司名
        if conversation_history:
            for hist in reversed(conversation_history):
                if isinstance(hist, dict) and hist.get("role") == "user":
                    hist_q = hist.get("content", "")
                    if hist_q and resolve_kb_name(hist_q) is not None:
                        return None

        # 生成澄清提示
        kbs = list_knowledge_bases()
        if not kbs:
            return None

        kb_list = "、".join(f'"{info["description"]}"' for _, info in kbs.items())
        return True, (
            f"请问您指的是哪家公司？当前可查询以下知识库：{kb_list}。"
            f"您可以直接说公司全称或简称。"
        )


_query_understanding_service = None


def get_query_understanding_service() -> QueryUnderstandingService:
    """
    作用：获取Query理解服务实例（单例）

    Returns:
        QueryUnderstandingService: 服务实例
    """
    global _query_understanding_service
    if _query_understanding_service is None:
        from .llm_router import get_llm_router
        try:
            llm_router = get_llm_router()
        except Exception:
            llm_router = None
        _query_understanding_service = QueryUnderstandingService(llm_router=llm_router)
    return _query_understanding_service

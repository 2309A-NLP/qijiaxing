"""
kb_mapping.py — 公司名 ↔ 知识库名称映射

作用：
  维护公司名称到 Milvus kb_name 的映射关系，供 Query 理解模块自动
  检测问题中提到的公司并填入 kb_name 过滤条件。

设计原理：
  用精确匹配 + 关键词匹配（缩写、简称、英文名）覆盖各种写法。
  新增公司只需在 KNOWLEDGE_BASE_MAP 中加一条记录，不修改任何业务代码。

用法：
  from app.core.kb_mapping import resolve_kb_name
  kb_name = resolve_kb_name("兴图新科")  # → "招股说明书1"

映射表格式：
  {kb_name: {full_names: [], keywords: []}}
  - full_names: 精确全称匹配（优先匹配）
  - keywords: 关键词/简称匹配（问题中包含任意一个就算命中）
"""

# =============================================================================
# 知识库映射定义
# =============================================================================
# 格式：{kb_name: {"full_names": [...], "keywords": [...]}}
# - full_names: 精确全称，匹配时优先
# - keywords: 模糊关键词，问题中包含任意一个即视为指向此知识库
# - priority: 优先级（数字越小越优先，0为最高）
KNOWLEDGE_BASE_MAP = {
    "招股说明书1": {
        "full_names": [
            "武汉兴图新科电子股份有限公司",
            "兴图新科电子股份有限公司",
        ],
        "keywords": [
            "兴图新科",
            "兴图",
            "Xingtuxinke",
            "视频指挥",
            "视频预警",
        ],
        "description": "武汉兴图新科 — 视频指挥控制/预警控制类产品",
        "priority": 1,
    },
    "招股说明书2": {
        "full_names": [
            "武汉力源信息技术股份有限公司",
            "力源信息技术股份有限公司",
            "武汉力源信息技术",
        ],
        "keywords": [
            "力源信息",
            "力源",
            "武汉力源",
            "P&S",
            "P&amp;S",
            "Mark Zhao",
            "赵马克",
            "Digi-Key",
            "目录销售商",
            "IC目录",
            "IC销售",
            "电子元器件",
        ],
        "description": "武汉力源信息 — IC目录销售商",
        "priority": 2,
    },
}


# =============================================================================
# 解析函数
# =============================================================================

def resolve_kb_name(question: str) -> str | None:
    """
    从问题中检测公司实体，返回对应的 kb_name。

    匹配策略（优先级从高到低）：
      1. 精确全称匹配（核对 full_names 列表）
      2. 关键词匹配（核对 keywords 列表，取优先级最高的命中）
      3. 无匹配 → 返回 None（不过滤）

    Args:
        question: 用户问题文本

    Returns:
        str | None: 匹配到的 kb_name，无匹配时返回 None

    面试官可能问：
      Q: 为什么不用 NLP 实体识别（如 jieba 词性标注）?
      A: 公司名识别在招股书场景下关键词足够覆盖。
         精确全称匹配防误伤（"力源"不会误匹配到其他公司），
         关键词匹配覆盖简称（"兴图"→兴图新科）。
         jieba 对机构名（nt）的识别在新上市公司名上准确率不高。

      Q: 如果问题同时提到了两家公司（如对比查询）怎么办？
      A: 同时命中多个时返回 priority 最高的那个。
         对比类查询本身不应该加 kb_name 过滤（需要跨库检索），
         这类问题会在意图识别环节被识别为 COMPARISON 类型，
         retrieval.py 对其特殊处理——即使有 detected_kb_name 也不过滤。
    """
    if not question or not question.strip():
        return None

    # 策略1：精确全称匹配
    for kb_name, info in KNOWLEDGE_BASE_MAP.items():
        for full_name in info["full_names"]:
            if full_name in question:
                return kb_name

    # 策略2：关键词匹配（按 priority 排序，取第一个命中）
    sorted_kbs = sorted(
        KNOWLEDGE_BASE_MAP.items(),
        key=lambda x: x[1].get("priority", 99),
    )
    for kb_name, info in sorted_kbs:
        for keyword in info["keywords"]:
            if keyword in question:
                return kb_name

    # 策略3：无匹配
    return None


def list_knowledge_bases() -> dict:
    """
    列出所有可用的知识库信息，用于调试和诊断。

    Returns:
        dict: {kb_name: {description, priority}}
    """
    return {
        kb_name: {
            "description": info["description"],
            "priority": info.get("priority", 99),
        }
        for kb_name, info in KNOWLEDGE_BASE_MAP.items()
    }

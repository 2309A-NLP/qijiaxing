#!/usr/bin/env python
"""
PyCharm AI助手 - 在PyCharm控制台中快速调用

功能：
1. 生成代码
2. 解释现有代码
3. 优化代码建议
4. 添加注释
5. 代码审查

使用方式：
1. 在PyCharm中右键打开Python控制台
2. 导入本模块
3. 使用 helper.generate_code() 等函数
"""

import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.llm_router import get_llm_router


class PyCharmAIHelper:
    """PyCharm AI助手"""

    def __init__(self):
        self.llm = get_llm_router()
        print("🚀 PyCharm AI助手已就绪！")

    async def generate_code(self, task: str) -> str:
        """
        生成代码

        Args:
            task: 任务描述
        """
        prompt = f"""你是一个专业的代码助手。请根据以下需求生成代码：

需求：{task}

要求：
- 代码结构清晰
- 包含必要的注释
- 遵循Python最佳实践
- 包含类型提示

请直接输出完整代码。"""

        print(f"📝 正在生成: {task}")
        result = await self.llm.generate(query=prompt, context=[])
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
        return result

    async def explain_code(self, code: str) -> str:
        """
        解释代码

        Args:
            code: 需要解释的代码
        """
        prompt = f"""请详细解释以下代码的功能和实现逻辑：

```python
{code}
```

请用中文解释，包括：
1. 整体功能概述
2. 关键逻辑说明
3. 使用场景"""

        print("🔍 正在解释代码...")
        result = await self.llm.generate(query=prompt, context=[])
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
        return result

    async def optimize_code(self, code: str) -> str:
        """
        优化代码建议

        Args:
            code: 需要优化的代码
        """
        prompt = f"""请分析以下代码并提出优化建议：

```python
{code}
```

请从以下方面分析：
1. 性能优化
2. 代码可读性
3. 最佳实践
4. 潜在bug

请提供优化后的代码。"""

        print("⚡ 正在优化代码...")
        result = await self.llm.generate(query=prompt, context=[])
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
        return result

    async def add_comments(self, code: str) -> str:
        """
        添加注释

        Args:
            code: 需要添加注释的代码
        """
        prompt = f"""请为以下代码添加详细的中文注释：

```python
{code}
```

要求：
- 添加函数/类的docstring
- 添加关键逻辑的行内注释
- 保持代码原有的功能不变
- 直接输出带注释的完整代码"""

        print("📝 正在添加注释...")
        result = await self.llm.generate(query=prompt, context=[])
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
        return result

    async def code_review(self, code: str) -> str:
        """
        代码审查

        Args:
            code: 需要审查的代码
        """
        prompt = f"""请对以下代码进行审查：

```python
{code}
```

审查方面：
1. 代码质量
2. 安全性
3. 潜在问题
4. 改进建议"""

        print("🔎 正在代码审查...")
        result = await self.llm.generate(query=prompt, context=[])
        print("\n" + "=" * 80)
        print(result)
        print("=" * 80)
        return result


# 创建全局实例
_helper = None


def get_helper() -> PyCharmAIHelper:
    """获取AI助手单例"""
    global _helper
    if _helper is None:
        _helper = PyCharmAIHelper()
    return _helper


# 快捷函数（方便在PyCharm控制台直接调用）
async def generate(task: str):
    """快捷：生成代码"""
    helper = get_helper()
    return await helper.generate_code(task)


async def explain(code: str):
    """快捷：解释代码"""
    helper = get_helper()
    return await helper.explain_code(code)


async def optimize(code: str):
    """快捷：优化代码"""
    helper = get_helper()
    return await helper.optimize_code(code)


async def comments(code: str):
    """快捷：添加注释"""
    helper = get_helper()
    return await helper.add_comments(code)


async def review(code: str):
    """快捷：代码审查"""
    helper = get_helper()
    return await helper.code_review(code)


print("""
╔════════════════════════════════════════════════════════════╗
║                    PyCharm AI 助手                        ║
╠════════════════════════════════════════════════════════════╣
║  使用方式:                                                  ║
║    1. 导入: from tools.pycharm_helper import *            ║
║    2. 调用: await generate("任务描述")                    ║
║                                                           ║
║  可用函数:                                                 ║
║    await generate("任务")    - 生成代码                    ║
║    await explain("代码")     - 解释代码                    ║
║    await optimize("代码")    - 优化建议                    ║
║    await comments("代码")    - 添加注释                    ║
║    await review("代码")      - 代码审查                    ║
╚════════════════════════════════════════════════════════════╝
""")

#!/usr/bin/env python
"""
自动写代码工具 - 利用项目LLM能力生成代码

使用方式:
    python tools/code_generator.py --task "实现一个排序算法"
    python tools/code_generator.py --file test.py --task "实现一个简单的计算器"
    python tools/code_generator.py --lang python --task "创建一个FastAPI接口"
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.core.llm_router import get_llm_router


CODE_GENERATOR_PROMPT = """你是一个专业的代码助手，擅长编写高质量、可维护的代码。

请根据用户需求生成代码，遵循以下要求：
1. 代码结构清晰，有适当的注释
2. 包含必要的错误处理
3. 遵循常见的编程最佳实践
4. 如果是Python代码，使用类型提示
5. 如果是Python代码，添加简单的示例用法

用户需求：{task}

请直接输出完整的代码，不要多余的解释。"""


async def generate_code(task: str, language: str = "python") -> str:
    """
    使用项目的LLM路由生成代码

    Args:
        task: 代码生成任务描述
        language: 目标编程语言

    Returns:
        生成的代码
    """
    llm = get_llm_router()

    prompt = CODE_GENERATOR_PROMPT.format(task=task)

    print(f"🚀 正在生成代码...")
    print(f"📋 任务: {task}")
    print(f"💻 语言: {language}")
    print("-" * 60)

    try:
        result = await llm.generate(query=prompt, context=[])
        return result
    except Exception as e:
        return f"生成失败: {str(e)}"


async def generate_and_save(task: str, file_path: str, language: str = "python"):
    """
    生成代码并保存到文件

    Args:
        task: 代码生成任务描述
        file_path: 保存文件路径
        language: 目标编程语言
    """
    code = await generate_code(task, language)

    if not code.startswith("生成失败"):
        file = Path(file_path)
        file.write_text(code, encoding="utf-8")
        print(f"\n✅ 代码已保存到: {file.absolute()}")
        print(f"\n📝 生成的代码:\n")
        print("=" * 60)
        print(code)
        print("=" * 60)
    else:
        print(code)


def main():
    parser = argparse.ArgumentParser(
        description="自动写代码工具 - 利用项目LLM能力生成代码"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="代码生成任务描述"
    )
    parser.add_argument(
        "--file", "-f",
        help="保存代码的文件路径（可选）"
    )
    parser.add_argument(
        "--lang", "-l",
        default="python",
        help="目标编程语言（默认: python）"
    )

    args = parser.parse_args()

    import asyncio
    if args.file:
        asyncio.run(generate_and_save(args.task, args.file, args.lang))
    else:
        code = asyncio.run(generate_code(args.task, args.lang))
        print("\n📝 生成的代码:\n")
        print("=" * 60)
        print(code)
        print("=" * 60)


if __name__ == "__main__":
    main()

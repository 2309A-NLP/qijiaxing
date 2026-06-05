#!/usr/bin/env python
"""
PyCharm外部工具配置脚本

此脚本生成可以在PyCharm中配置的外部工具命令，
让你可以右键点击文件或选中文本直接调用AI功能！
"""

from pathlib import Path


def print_pycharm_config():
    """打印PyCharm外部工具配置"""

    config = """
╔═══════════════════════════════════════════════════════════════╗
║         PyCharm 外部工具配置指南                                ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  配置步骤:                                                     ║
║    1. 打开 PyCharm                                             ║
║    2. File → Settings → Tools → External Tools                ║
║    3. 点击 + 号，添加以下工具                                  ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  工具1: AI 代码生成 (AI Code Generator)                       ║
║  ──────────────────────────────────────────────────────────────║
║  Name:              AI Code Generator                         ║
║  Program:           python                                    ║
║  Arguments:         tools/code_generator.py --task "$SelectedText$" --file "$FilePath$"  ║
║  Working directory: $ProjectFileDir$/backend                  ║
║                                                               ║
║  工具2: AI 解释选中代码 (AI Explain Code)                     ║
║  ──────────────────────────────────────────────────────────────║
║  Name:              AI Explain Code                           ║
║  Program:           python                                    ║
║  Arguments:         -c "from tools.pycharm_helper import explain; import asyncio; asyncio.run(explain('''$SelectedText$'''))"  ║
║  Working directory: $ProjectFileDir$/backend                  ║
║                                                               ║
║  工具3: AI 优化选中代码 (AI Optimize Code)                    ║
║  ──────────────────────────────────────────────────────────────║
║  Name:              AI Optimize Code                          ║
║  Program:           python                                    ║
║  Arguments:         -c "from tools.pycharm_helper import optimize; import asyncio; asyncio.run(optimize('''$SelectedText$'''))"  ║
║  Working directory: $ProjectFileDir$/backend                  ║
║                                                               ║
║  工具4: AI 添加注释 (AI Add Comments)                        ║
║  ──────────────────────────────────────────────────────────────║
║  Name:              AI Add Comments                           ║
║  Program:           python                                    ║
║  Arguments:         -c "from tools.pycharm_helper import comments; import asyncio; asyncio.run(comments('''$SelectedText$'''))"  ║
║  Working directory: $ProjectFileDir$/backend                  ║
║                                                               ║
║  工具5: AI 代码审查 (AI Code Review)                          ║
║  ──────────────────────────────────────────────────────────────║
║  Name:              AI Code Review                            ║
║  Program:           python                                    ║
║  Arguments:         -c "from tools.pycharm_helper import review; import asyncio; asyncio.run(review('''$SelectedText$'''))"  ║
║  Working directory: $ProjectFileDir$/backend                  ║
║                                                               ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  使用方式:                                                     ║
║    - 在编辑器中选中代码                                         ║
║    - 右键 → External Tools → 选择工具                          ║
║    - 或使用快捷键 (需要在 Keymap 中配置)                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(config)


def create_quick_scripts():
    """创建快速脚本"""

    scripts_dir = Path(__file__).parent / "quick_scripts"
    scripts_dir.mkdir(exist_ok=True)

    # 脚本1: 生成代码
    (scripts_dir / "quick_gen.py").write_text("""
from tools.pycharm_helper import generate
import asyncio
import sys

if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = input("请输入代码生成任务: ")

asyncio.run(generate(task))
""", encoding="utf-8")

    # 脚本2: 解释代码
    (scripts_dir / "quick_explain.py").write_text("""
from tools.pycharm_helper import explain
import asyncio
import sys

if len(sys.argv) > 1:
    code = sys.argv[1]
else:
    code = input("请输入要解释的代码: ")

asyncio.run(explain(code))
""", encoding="utf-8")

    print(f"✅ 快速脚本已创建到: {scripts_dir}")


if __name__ == "__main__":
    print_pycharm_config()
    create_quick_scripts()

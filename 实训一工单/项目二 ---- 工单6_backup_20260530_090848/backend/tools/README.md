# 🛠️ 开发工具集合

## 工具箱概览

| 工具 | 文件 | 功能 |
|------|------|------|
| 代码生成器 | [code_generator.py](code_generator.py) | 命令行代码生成工具 |
| PyCharm助手 | [pycharm_helper.py](pycharm_helper.py) | PyCharm控制台AI助手 |
| 配置脚本 | [setup_pycharm_tools.py](setup_pycharm_tools.py) | PyCharm外部工具配置指南 |

---

## code_generator.py - 自动写代码工具

利用项目的LLM能力自动生成代码！

### 🚀 快速开始

#### 基本用法（直接输出到终端）
```bash
cd backend
python tools/code_generator.py --task "实现一个快速排序算法"
```

#### 保存到文件
```bash
python tools/code_generator.py --task "创建一个简单的计算器" --file calculator.py
```

#### 指定编程语言
```bash
python tools/code_generator.py --task "创建一个React组件" --lang javascript --file MyComponent.jsx
```

### 📋 命令行参数

| 参数 | 简写 | 说明 | 必须 |
|------|------|------|------|
| `--task` | `-t` | 代码生成任务描述 | ✅ 是 |
| `--file` | `-f` | 保存文件路径（可选） | ❌ 否 |
| `--lang` | `-l` | 目标编程语言（默认: python） | ❌ 否 |

---

## pycharm_helper.py - PyCharm AI助手

在PyCharm Python控制台中使用的AI工具！

### 🔥 快速使用

#### 在PyCharm控制台中：
```python
# 1. 导入
from tools.pycharm_helper import *

# 2. 使用
await generate("实现一个二分查找算法")
await explain("""def foo(x): return x * 2""")
await optimize("some code here")
await comments("code here")
await review("code here")
```

### 📦 可用功能

| 函数 | 功能 |
|------|------|
| `await generate("任务")` | 生成代码 |
| `await explain("代码")` | 解释代码 |
| `await optimize("代码")` | 优化建议 |
| `await comments("代码")` | 添加注释 |
| `await review("代码")` | 代码审查 |

---

## setup_pycharm_tools.py - 配置PyCharm外部工具

运行此脚本查看如何在PyCharm中配置右键菜单AI工具：

```bash
python tools/setup_pycharm_tools.py
```

### 配置的工具包括：
- AI Code Generator - 生成代码到当前文件
- AI Explain Code - 解释选中代码
- AI Optimize Code - 优化选中代码
- AI Add Comments - 添加注释
- AI Code Review - 代码审查

---

### ⚙️ 配置要求

确保项目的LLM服务配置正确：
- Ollama本地模型
- 或DeepSeek API（在`app/config.py`中配置）
- 或SGLang服务

### 📝 生成质量提示

描述任务时越具体，生成的代码质量越高：

| 好的任务描述 | 不好的描述 |
|------------|-----------|
| "实现一个Python的LRU缓存类，带过期时间功能" | "写个缓存" |
| "创建一个FastAPI的文件上传接口，支持jpg/png，最大10MB" | "写个上传接口" |
| "实现一个React的登录表单组件，带表单验证" | "写个登录页" |

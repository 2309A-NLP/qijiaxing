"""
parse_pdf_mineru.py — MinerU版PDF解析脚本
=============================================
作用：使用MinerU的pipeline后端解析招股说明书PDF，提取文字内容
      输出格式与原始parse_pdf.py兼容（list[dict]），下游chunk_text.py等无需改动

原理：
  1. MinerU的pipeline后端比PyMuPDF(fitz)多了版面分析功能：
     - 自动检测多栏排版，按正确阅读顺序排列
     - 自动丢弃页眉、页脚、页码（通过page_header/page_footer/page_number类型标记）
     - 自动识别公式（LaTeX输出）
     - 自动提取表格（HTML格式）
     - 扫描件自动切换OCR
  2. 使用CONTENT_LIST_V2输出格式，按页分组的结构化内容
  3. 每页的内容块转成与原始parse_pdf.py一致的 dict 格式

对比原始方案（parse_pdf.py）：
  原始：fitz.get_text()直接提取文本 → 含页眉页码 → 下游chunk_text.py手动过滤
  MinerU：pipeline后端自动做完版面分析+内容提取+去噪 → 直接得到干净文本

输入：data/招股说明书1.pdf
输出：data/parsed/招股说明书_原始文本.json（与原parse_pdf.py输出路径一致）

依赖：
  - mineru>=3.1.0（含pipeline后端）
  - pipeline模型：opendatalab/PDF-Extract-Kit-1.0
  - 运行前设置环境变量：MINERU_MODEL_SOURCE=modelscope（或huggingface）
"""

import json
import re
import shutil
import sys
import os
from pathlib import Path

# MinerU在8GB系统RAM上默认window_size=64会导致OOM，限制为4页
os.environ.setdefault("MINERU_PROCESSING_WINDOW_SIZE", "4")

# =============================================================================
# 模块路径设置
# =============================================================================
# 将项目根目录添加到Python模块搜索路径，确保能导入config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    get_kb_paths,  # v2.0: 获取指定KB的文件路径
    PARSED_DIR,
    RUN_TIMESTAMP,
    setup_logger,
    VLM_CONFIG_PATH,  # VLM配置路径（用于图表理解）
)

from mineru.cli.common import do_parse, read_fn
from mineru.utils.enum_class import MakeMode

# 创建日志记录器
logger = setup_logger("parse_pdf_mineru")

# =============================================================================
# 图表OCR配置（使用MinerU内置PytorchPaddleOCR）
# =============================================================================
# OCR实例（懒加载，首次使用时初始化）
_ocr_instance = None
# MinerU输出目录（chart图片的相对路径基于此目录）
_mineru_output_dir: str = ""


# =============================================================================
# 常量
# =============================================================================
# MinerU content_list_v2中需要丢弃的块类型（非正文元素）
# page_header: 页眉（如"武汉兴图新科电子股份有限公司  招股意向书"）
# page_footer: 页脚
# page_number: 页码（如"1-1-1"）
DISCARD_BLOCK_TYPES = {"page_header", "page_footer", "page_number"}

# OCR水印过滤模式（PDF评估版/演示版水印，干扰图表OCR识别）
# 这些文本是叠加在PDF图片上的水印文字，OCR会误识别为图表内容
# 格式：列表中的字符串，匹配任一即过滤整段
WATERMARK_PATTERNS = [
    "八维教育", "刘敏", "人工智能刘敏", "工智能刘敏",
    "lium", "engliu", "ghinen", "gliumin", "hinengliumin",
    "rengongzhineng", "nengzhineng", "zhinengliumin",
    "武汉八维", "八维", "维教育",
]


# =============================================================================
# OCR水印过滤函数
# =============================================================================
def _clean_ocr_text(text: str) -> str:
    """
    作用：过滤OCR结果中的水印干扰文本
    
    原理：
      PDF评估版的水印文字（如"八维教育"）会被OCR误识别为图表内容，混入有效数据中。
      水印通常出现在OCR输出的末尾，且包含固定中文短语+乱码英文的组合模式。
    
    逻辑：
      1. 按空格/换行分割为单词片段
      2. 过滤掉包含已知水印关键词的片段
      3. 过滤纯英文小写且长度>5的片段（水印特征：rengongzhineng等）
      4. 重组剩余片段
    
    参数：
      text — OCR原始输出文本
    
    返回：
      过滤后的文本
    """
    import re
    if not text:
        return ""
    
    segments = re.split(r'[\s\n]+', text)
    clean_segments = []
    
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # 检查水印关键词
        is_watermark = False
        for pattern in WATERMARK_PATTERNS:
            if pattern.lower() in seg.lower():
                is_watermark = True
                break
        if is_watermark:
            continue
        # 过滤纯英文小写长字符串（水印特征：英文乱码）
        if seg.isascii() and seg.islower() and len(seg) > 4:
            continue
        # 过滤仅包含标点和数字的无效片段
        if all(c in '0123456789%.,:;()[]-+ ' for c in seg):
            clean_segments.append(seg)
        else:
            clean_segments.append(seg)
    
    return " ".join(clean_segments)


# =============================================================================
# VLM图表理解函数（使用视觉语言模型理解图表结构）
# =============================================================================
def _describe_chart_with_vlm(image_path: str) -> str:
    """
    作用：使用VLM视觉语言模型理解图表图片，返回结构化描述

    原理：
      纯OCR只能提取图片中的文字，无法理解图表的视觉结构（柱状图高低对比、
      折线图趋势、饼图占比等）。VLM可以"看图"，识别坐标轴、数据点、趋势等。

    参数：
      image_path — 图表图片的绝对路径

    返回：
      VLM对图表的结构化描述文本；失败时返回空字符串

    边界情况：
      - 图片不存在：返回空字符串
      - API调用失败/超时：返回空字符串（不中断解析流程）
      - VLM不支持图片：返回空字符串
    """
    import base64
    import json
    import urllib.request
    import urllib.error

    img_file = Path(image_path)
    if not img_file.exists():
        logger.warning(f"VLM图表理解：图片不存在 {image_path}")
        return ""

    # 读取VLM配置
    try:
        with open(VLM_CONFIG_PATH, "r", encoding="utf-8") as f:
            vlm_cfg = json.load(f)
    except Exception as e:
        logger.warning(f"VLM配置文件读取失败: {e}")
        return ""

    api_host = vlm_cfg.get("api_host", "")
    api_key = vlm_cfg.get("api_key", "")
    model = vlm_cfg.get("model", "Mimo-v2.5-pro")
    max_tokens = vlm_cfg.get("max_tokens", 1024)
    temperature = vlm_cfg.get("temperature", 0.1)

    if not api_host or not api_key:
        logger.warning("VLM配置缺少api_host或api_key")
        return ""

    # Base64编码图片
    try:
        with open(str(img_file), "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning(f"VLM图表理解：图片读取失败 {e}")
        return ""

    # 构造请求
    prompt = (
        "你是一个专业的图表分析专家。请详细描述这张图表的所有数据和结构信息。"
        "要求：\n"
        "1. 提取图表的标题、坐标轴标签\n"
        "2. 列出所有数据点及其对应的数值\n"
        "3. 描述数据的整体趋势（上升/下降/波动）、变化幅度\n"
        "4. 如果有多个数据系列，比较它们之间的关系（大/小、快/慢等）\n"
        "5. 注意：只输出图表中的实际数据，不要添加编造的数据\n"
        "6. 如果图表的文字模糊无法识别，如实说明\n\n"
        "请按以下格式输出：\n"
        "【图表标题】xxx\n"
        "【数据点】xxx: 值1, xxx: 值2, ...\n"
        "【趋势分析】xxx\n"
        "【数据关系】xxx"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ]

    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    # 发送请求
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    url = api_host.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            description = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if description:
                logger.info(f"VLM图表理解成功：{Path(image_path).name}，描述 {len(description)} 字符")
                return description.strip()
            else:
                logger.warning(f"VLM图表理解：返回空内容")
                return ""
    except urllib.error.HTTPError as e:
        logger.warning(f"VLM图表理解HTTP错误 {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}")
        return ""
    except Exception as e:
        logger.warning(f"VLM图表理解失败: {e}")
        return ""


# =============================================================================
def backup_previous_output(output_path: str) -> None:
    """
    作用：备份旧输出文件，用于回滚
    原理：每次运行可能覆盖旧结果，备份后如果新结果有问题可以恢复
    参数：output_path — 输出文件路径
    """
    path = Path(output_path)
    if path.exists():
        backup_name = f"{path.stem}_备份_{RUN_TIMESTAMP}{path.suffix}"
        backup_path = path.parent / backup_name
        shutil.copy2(path, backup_path)
        logger.info(f"已备份旧文件 -> {backup_path}")


# =============================================================================
# MinerU内容块文本提取
# =============================================================================
def _extract_text_recursive(content) -> str:
    """
    作用：递归提取MinerU内容块中的纯文本

    原理：
      MinerU的CONTENT_LIST_V2格式采用嵌套结构，例如段落块：
        {"type":"paragraph", "content":{"paragraph_content":[{"type":"text","content":"..."}]}}
      需要递归遍历所有层级的 "content" 字段提取字符串

    参数：
      content — 任何类型的嵌套数据（dict、list、str）

    返回：
      提取到的纯文本字符串，层级间以空格连接

    边界情况：
      - content为空/None：返回空字符串
      - content直接是字符串：直接返回
      - content为dict：遍历'content'/'paragraph_content'/'title_content'/'item_content'等键
      - content为list：递归处理每个元素
    """
    texts = []

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # 直接有content字段
        if "content" in content:
            texts.append(_extract_text_recursive(content["content"]))
        # 段落内容：paragraph_content → [{type, content}, ...]
        if "paragraph_content" in content:
            for item in content["paragraph_content"]:
                texts.append(_extract_text_recursive(item))
        # 标题内容：title_content → [{type, content}, ...]
        if "title_content" in content:
            for item in content["title_content"]:
                texts.append(_extract_text_recursive(item))
        # 列表项内容：item_content → [{type, content}, ...]
        if "item_content" in content:
            for item in content["item_content"]:
                texts.append(_extract_text_recursive(item))
        # 列表项：list_items → [{item_type, item_content}, ...]
        if "list_items" in content:
            for item in content["list_items"]:
                if "item_content" in item:
                    for sub in item["item_content"]:
                        texts.append(_extract_text_recursive(sub))

    elif isinstance(content, list):
        for item in content:
            texts.append(_extract_text_recursive(item))

    return " ".join(filter(None, texts))


# =============================================================================
# 图表OCR识别函数
# =============================================================================
def _get_ocr_instance():
    """
    作用：懒加载MinerU内置的PytorchPaddleOCR实例
    
    原理：
      PytorchPaddleOCR是MinerU封装的OCR引擎，内部使用pytorchocr模型（PaddleOCR的PyTorch移植版）。
      首次调用时初始化（下载模型+加载权重），后续复用同一实例。
      
    返回：
      PytorchPaddleOCR实例
    """
    global _ocr_instance
    if _ocr_instance is None:
        from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
        logger.info("初始化MinerU OCR引擎（首次使用，加载模型...）")
        _ocr_instance = PytorchPaddleOCR(lang='ch')
        logger.info("MinerU OCR引擎初始化完成")
    return _ocr_instance


def _extract_chart_text_with_ocr(image_path: str) -> str:
    """
    作用：用MinerU内置OCR识别图表图片中的文字

    原理：
      1. 用cv2读取图片
      2. 调用PytorchPaddleOCR的ocr()方法做文字检测+识别
      3. 将识别结果拼接为文本

    参数：
      image_path — 图片文件的绝对路径

    返回：
      图表中的文字内容；失败时返回空字符串

    边界情况：
      - 图片文件不存在：返回空字符串
      - OCR未识别到文字：返回空字符串
      - 初始化失败：返回空字符串（不中断解析流程）
    """
    import cv2
    import numpy as np

    img_file = Path(image_path)
    if not img_file.exists():
        logger.warning(f"图表图片不存在: {image_path}")
        return ""

    try:
        ocr = _get_ocr_instance()
        # cv2.imread不支持中文路径，用numpy读取绕过
        img_bytes = np.fromfile(str(img_file), dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"图片读取失败: {image_path}")
            return ""

        ocr_res = ocr.ocr(img, det=True, rec=True)
        if not ocr_res or not ocr_res[0]:
            return ""

        # ocr_res格式: [[[box, (text, score)], ...]]
        texts = []
        for item in ocr_res[0]:
            if item and len(item) == 2:
                text, score = item[1]
                if score >= 0.5:  # 过滤低置信度
                    texts.append(text)

        result = " ".join(texts)
        # 水印过滤：去掉PDF评估版水印的干扰文字
        result = _clean_ocr_text(result)
        if result:
            logger.info(f"图表OCR识别成功，提取 {len(result)} 字符（水印过滤后）")
        return result

    except Exception as e:
        logger.warning(f"图表OCR识别失败: {e}")
        return ""


def block_to_text(block: dict) -> str | None:
    """
    作用：将MinerU的单个内容块转换为纯文本

    参数：
      block — MinerU内容块dict，格式如：
        {"type":"paragraph", "content":{...}, "bbox":[x0,y0,x1,y1]}

    返回：
      纯文本字符串；如果块应该被丢弃（页眉/页脚/页码）则返回None

    块类型处理策略：
      - page_header/page_footer/page_number → 丢弃（返回None）
      - table → 返回HTML格式（如有）或None（图片表格）
      - image/chart/seal → 跳过（返回None，无可用文本）
      - paragraph/title/list/index → 递归提取文本
    """
    block_type = block.get("type", "")
    content = block.get("content", {})

    # 丢弃非正文元素
    if block_type in DISCARD_BLOCK_TYPES:
        return None

    # 表格：有HTML则返回HTML，否则跳过
    if block_type == "table":
        html = content.get("html", "")
        if html:
            return html
        return None

    # 图表：优先使用VLM理解图表结构，VLM失败则回退到OCR+水印过滤
    if block_type == "chart":
        parts = []
        # 1. 优先使用VLM理解图表结构
        image_source = content.get("image_source", {})
        rel_path = image_source.get("path", "")
        if rel_path and _mineru_output_dir:
            abs_path = os.path.join(_mineru_output_dir, rel_path)
            vlm_text = _describe_chart_with_vlm(abs_path)
            if vlm_text:
                parts.append(f"[VLM图表描述] {vlm_text}")
                logger.info(f"图表 {Path(abs_path).name}: 使用VLM理解")
            else:
                # VLM失败，回退到OCR+水印过滤
                ocr_text = _extract_chart_text_with_ocr(abs_path)
                if ocr_text:
                    parts.append(f"[OCR图表文字] {ocr_text}")
                    logger.info(f"图表 {Path(abs_path).name}: VLM失败，使用OCR")
        # 2. 提取footnote（图表的附注文字）
        footnotes = content.get("img_footnote", [])
        for fn in footnotes:
            fn_text = _extract_text_recursive(fn)
            if fn_text:
                parts.append(f"[图表附注] {fn_text}")
        if parts:
            return " ".join(parts)
        return None

    # 图片/印章：无法提取文字
    if block_type in ("image", "seal"):
        return None

    # 正文元素：段落、标题、列表、索引等
    text = _extract_text_recursive(content)
    return text.strip() if text else None


def extract_page_text_from_content_list_v2(
    content_list_v2: list,
    source_pdf: str,
) -> list[dict]:
    """
    作用：将MinerU的CONTENT_LIST_V2输出转换为项目二的标准解析格式

    MinerU输出格式：
      [
        [block_1_1, block_1_2, ...],  # 第1页的内容块列表
        [block_2_1, block_2_2, ...],  # 第2页的内容块列表
        ...
      ]
      每个block：
        {"type": "paragraph"|"title"|"table"|...,
         "content": {...},
         "bbox": [x0, y0, x1, y1]}

    标准输出格式：
      [
        {"page_num": 1, "source_pdf": "...", "text": "...",
         "char_count": N, "type": "text", "section": "第一节 释义"},
        ...
      ]

    参数：
      content_list_v2 — MinerU输出的按页分组的内容列表
      source_pdf — PDF源文件路径

    返回：
      标准格式的解析结果列表

    设计说明：
      - 每页的所有有效内容块合并为一个"text"条目（保持与原始格式兼容）
      - 页眉/页脚/页码被MinerU自动识别并丢弃
      - 空页跳过不添加记录
      - char_count用于质量检查和统计
      - section字段通过正则匹配页面文本中的章节标题提取

    面试官可能问：
      Q: 为什么每页只输出一个text条目，而不是每个内容块独立？
      A: 原始parse_pdf.py的输出格式是"每页一条"，下游chunk_text.py按此格式
         设计。保持兼容性可以减少改动范围。内容块的分割由chunk阶段完成。

      Q: 图片类的表格（html为空）怎么处理？
      A: 跳过。图片表格没有可提取的文本内容，需要OCR才能提取。
         如果要处理图片表格，需要启用pipeline的table功能，或使用hybrid后端。

      Q: 丢弃块类型的判断依据是什么？
      A: MinerU的版面分析模型会为每个内容块标注类型。page_header/page_footer/
         page_number是模型自动判断的，不需要手动配置规则。
    """
    # 章节标题正则：匹配"第X节 XXX"格式（允许节号和标题之间有空格）
    # 示例：第一节 释义、第二节 概览、第三节 本次发行概况
    SECTION_PATTERN = re.compile(r"^第[一二三四五六七八九十百]+节\s+\S+")

    current_section = None  # 当前章节名，跨页继承
    pages = []

    for page_idx, page_contents in enumerate(content_list_v2):
        page_num = page_idx + 1  # 页码从1开始（用户友好）

        if not page_contents:
            logger.debug(f"第 {page_num} 页无有效内容，跳过")
            continue

        # 分离文本块、表格块和图表块
        # 表格和图表单独输出为独立条目，确保下游用对应分块策略
        page_texts = []
        for block in page_contents:
            block_type = block.get("type", "")
            text = block_to_text(block)
            if not text:
                continue

            # 检测章节标题（仅从title或paragraph类型的块中检测）
            if block_type in ("title", "paragraph"):
                # 取文本前100字符检测，避免长段落误匹配
                first_line = text.strip().split("\n")[0].strip()
                m = SECTION_PATTERN.match(first_line)
                if m:
                    current_section = m.group(0)
                    logger.debug(f"第 {page_num} 页检测到章节: {current_section}")

            # 表格：单独输出，不与文本合并
            if block_type == "table":
                pages.append({
                    "page_num": page_num,
                    "source_pdf": source_pdf,
                    "text": text,
                    "char_count": len(text),
                    "type": "table",
                    "section": current_section,
                })
            # 图表：单独输出，不与文本合并
            elif block_type == "chart":
                pages.append({
                    "page_num": page_num,
                    "source_pdf": source_pdf,
                    "text": text,
                    "char_count": len(text),
                    "type": "chart",
                    "section": current_section,
                })
            else:
                page_texts.append(text)

        # 文本块：合并为一个条目
        if page_texts:
            combined = "\n".join(page_texts)
            pages.append({
                "page_num": page_num,
                "source_pdf": source_pdf,
                "text": combined,
                "char_count": len(combined),
                "type": "text",
                "section": current_section,
            })

    return pages


# =============================================================================
# MinerU解析主函数
# =============================================================================
def parse_pdf_with_mineru(
    pdf_path: str,
    output_dir: str,
) -> list[dict]:
    """
    作用：使用MinerU pipeline后端解析PDF，返回标准格式的内容列表

    原理：
      1. 调用do_parse()执行MinerU解析，输出到指定目录
      2. 从输出目录读取CONTENT_LIST_V2格式的结果文件
      3. 转换为本项目标准格式

    参数：
      pdf_path — PDF文件的绝对路径
      output_dir — MinerU输出目录

    返回：
      标准格式的解析结果列表
      每条记录包含：page_num, source_pdf, text, char_count, type

    API调用链：
      read_fn(pdf_path) → PDF字节
      → do_parse(output_dir, pdf_bytes, ...) → MinerU核心解析
        → _process_pipeline() → pipeline后端处理
          → MineruPipelineModel() → 加载模型(Layout+OCR+Formula+Table)
          → pipeline_doc_analyze_streaming() → 滑动窗口推理
          → union_make(pdf_info, CONTENT_LIST_V2) → 生成结构化JSON
      → 读取 *_content_list_v2.json → 格式转换

    注意事项：
      - formula_enable=True时加载UniMERNet公式识别模型（~773MB）
      - table_enable=True时加载StructEqTable表格模型（~1.75GB）
      - 环境变量MINERU_MODEL_SOURCE控制模型下载源
      - 首次运行会自动下载缺失的模型文件

    边界情况：
      - 解析失败时抛出异常，由调用方处理
      - 空PDF（0页）返回空列表
      - 扫描件PDF会自动启用OCR
    """
    pdf_name = Path(pdf_path).stem  # 文件名（不含扩展名）
    parse_method = "auto"  # auto：自动检测扫描件/文字版

    logger.info(f"开始MinerU解析: {pdf_path}")

    # -------------------------------------------------------------------------
    # 步骤1：读取PDF文件为字节数据
    # -------------------------------------------------------------------------
    pdf_bytes = read_fn(Path(pdf_path))

    # -------------------------------------------------------------------------
    # 步骤2：调用MinerU解析
    # -------------------------------------------------------------------------
    # do_parse参数说明：
    #   output_dir — 输出目录
    #   pdf_file_names — PDF文件名列表（不含路径）
    #   pdf_bytes_list — PDF字节数据列表
    #   p_lang_list — 语言列表，["ch"]表示中文
    #   backend="pipeline" — 使用pipeline后端（不需要VLM模型）
    #   parse_method="auto" — 自动检测扫描件/文字版
    #   formula_enable=True — 启用公式识别（LaTeX输出）
    #   table_enable=True — 启用表格识别（HTML输出）
    #   其余f_dump_*参数控制输出文件类型（为减少磁盘占用只保留content_list）
    do_parse(
        output_dir=output_dir,
        pdf_file_names=[pdf_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=["ch"],
        backend="pipeline",
        parse_method=parse_method,
        formula_enable=True,
        table_enable=True,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.CONTENT_LIST_V2,
    )

    # -------------------------------------------------------------------------
    # 步骤3：读取解析结果
    # -------------------------------------------------------------------------
    # MinerU输出目录结构：
    #   output_dir/{pdf_name}/{parse_method}/{pdf_name}_content_list_v2.json
    result_dir = Path(output_dir) / pdf_name / parse_method
    result_file = result_dir / f"{pdf_name}_content_list_v2.json"

    if not result_file.exists():
        logger.error(f"MinerU解析结果文件不存在: {result_file}")
        # 回退：尝试读取content_list（V1版本）
        result_file = result_dir / f"{pdf_name}_content_list.json"
        if not result_file.exists():
            raise FileNotFoundError(f"MinerU解析未生成结果文件: {result_file}")

    logger.info(f"读取解析结果: {result_file}")
    with open(result_file, "r", encoding="utf-8") as f:
        content_list_v2 = json.load(f)

    # 设置MinerU输出目录（图表图片的相对路径基于此目录）
    global _mineru_output_dir
    _mineru_output_dir = str(result_dir)

    # -------------------------------------------------------------------------
    # 步骤4：转换为标准格式
    # -------------------------------------------------------------------------
    source_pdf = str(Path(pdf_path).resolve())
    pages = extract_page_text_from_content_list_v2(content_list_v2, source_pdf)

    # -------------------------------------------------------------------------
    # 统计信息
    # -------------------------------------------------------------------------
    total_blocks = sum(len(page) for page in content_list_v2)
    discarded = sum(
        1 for page in content_list_v2 for b in page
        if b.get("type") in DISCARD_BLOCK_TYPES
    )

    logger.info(
        f"MinerU解析完成: {len(pages)} 个内容块 "
        f"（{pdf_name}，共 {len(content_list_v2)} 页，"
        f"{total_blocks} 个原始块，已丢弃 {discarded} 个非正文块）"
    )

    return pages


# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    """
    执行流程：
      1. 检查PDF文件是否存在
      2. 备份旧解析结果
      3. 使用MinerU解析PDF
      4. 保存解析结果为JSON（兼容原格式）
      5. 输出统计信息

    使用方式：
      # 先设置模型下载源
      export MINERU_MODEL_SOURCE=modelscope  # 或 huggingface
      export MINERU_PDF_RENDER_THREADS=1     # WSL环境需要

      # 运行解析
      python offline/parse_pdf_mineru.py

    与其他脚本的配合：
      - 输出文件被 chunk_text.py 读取进行分块
      - 分块结果被 generate_embeddings.py 向量化
      - 向量被导入 Milvus 供在线服务检索

    注意事项：
      - 首次运行会自动下载pipeline模型（约2-3GB）
    注意事项：
      - 如果遇到BrokenProcessPool错误，设置MINERU_PDF_RENDER_THREADS=1
    """
    # v2.0: 确定知识库名称
    import os, sys as _sys
    kb_name = _sys.argv[1] if len(_sys.argv) > 1 else os.environ.get("KB_NAME", "招股说明书2")
    kb_paths = get_kb_paths(kb_name)
    pdf_path = str(kb_paths["pdf_path"])
    parsed_json_path = kb_paths["parsed_json_path"]

    if not Path(pdf_path).exists():
        logger.error(f"PDF文件不存在: {pdf_path}")
        sys.exit(1)

    # 解析输出目录
    output_dir = str(PARSED_DIR)

    # 备份旧输出
    backup_previous_output(str(parsed_json_path))

    # 执行MinerU解析
    pages = parse_pdf_with_mineru(pdf_path, output_dir)

    # 保存结果
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(parsed_json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    logger.info(f"解析结果已保存: {parsed_json_path} ({len(pages)} 条)")
    print(f"\nMinerU解析完成！共 {len(pages)} 个内容块")
    print(f"输出文件: {parsed_json_path}")

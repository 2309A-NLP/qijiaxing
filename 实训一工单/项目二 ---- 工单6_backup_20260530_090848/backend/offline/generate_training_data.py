"""
生成 BGE-M3 微调训练数据
从 Milvus 提取文档块，用 DeepSeek 生成对应的问题
运行：D:\an\envs\project2\python.exe offline/generate_training_data.py
"""
import sys, os, json, time
sys.path.insert(0, ".")

from pymilvus import MilvusClient
from app.config import settings
import requests

# DeepSeek API 配置
DEEPSEEK_HOST = settings.DEEPSEEK_HOST
DEEPSEEK_KEY = settings.DEEPSEEK_KEY
DEEPSEEK_MODEL = settings.DEEPSEEK_MODEL

# 生成问题的 prompt
QUESTION_PROMPT = """你是一个专业的文档分析师。请根据以下文档内容，生成一个用户可能会问的问题。

要求：
1. 问题要自然、口语化，像真实用户会问的
2. 问题要能从这段文档中找到答案
3. 问题要具体，不要太宽泛
4. 只输出问题，不要输出其他内容

文档内容：
{content}

问题："""

def generate_question(content: str, max_retries: int = 3) -> str:
    """用 DeepSeek API 生成问题"""
    url = f"{DEEPSEEK_HOST}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json"
    }
    
    # 截断过长的内容
    if len(content) > 1500:
        content = content[:1500] + "..."
    
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": QUESTION_PROMPT.format(content=content)}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                question = result["choices"][0]["message"]["content"].strip()
                # 清理可能的前缀
                if question.startswith("问题："):
                    question = question[3:].strip()
                if question.startswith("Q：") or question.startswith("Q:"):
                    question = question[2:].strip()
                return question
            else:
                print(f"  API 错误 {resp.status_code}: {resp.text[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  请求异常: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return ""

def main():
    # 连接 Milvus
    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    client = MilvusClient(uri=uri)
    collection = settings.MILVUS_COLLECTION
    
    print(f"连接 Milvus: {uri}")
    print(f"集合: {collection}")
    
    # 查询所有文档块
    print("正在查询文档块数量...")
    # 先获取总数
    stats = client.get_collection_stats(collection_name=collection)
    total_count = int(stats.get("row_count", 0))
    print(f"文档块总数: {total_count}")
    
    if total_count == 0:
        print("错误：Milvus 中没有数据")
        return
    
    # 分批查询所有文档块
    batch_size = 100
    all_chunks = []
    
    print(f"正在提取文档块（批次大小: {batch_size}）...")
    for offset in range(0, total_count, batch_size):
        try:
            # 使用 query 方法分页查询
            results = client.query(
                collection_name=collection,
                filter="",  # 无过滤条件
                output_fields=["chunk_id", "text", "section", "content_type", "kb_name"],
                limit=batch_size,
                offset=offset
            )
            if results:
                all_chunks.extend(results)
                print(f"  已提取 {len(all_chunks)}/{total_count}")
        except Exception as e:
            print(f"  查询失败 (offset={offset}): {e}")
            # 尝试不使用 offset
            try:
                results = client.query(
                    collection_name=collection,
                    filter="",
                    output_fields=["chunk_id", "text", "section", "content_type", "kb_name"],
                    limit=batch_size
                )
                if results:
                    all_chunks.extend(results)
            except Exception as e2:
                print(f"  重试失败: {e2}")
        
        time.sleep(0.1)  # 避免请求过快
    
    # 去重
    seen_ids = set()
    unique_chunks = []
    for chunk in all_chunks:
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)
    
    all_chunks = unique_chunks
    print(f"提取完成，去重后共 {len(all_chunks)} 个文档块")
    
    if not all_chunks:
        print("错误：未提取到文档块")
        return
    
    # 检查 API 配置
    if not DEEPSEEK_KEY:
        print("错误：DEEPSEEK_KEY 未配置，请检查 .env 文件")
        return
    
    print(f"\n开始生成训练数据...")
    print(f"API: {DEEPSEEK_HOST}")
    print(f"模型: {DEEPSEEK_MODEL}")
    
    # 生成训练数据
    training_data = []
    failed_count = 0
    
    for i, chunk in enumerate(all_chunks):
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        section = chunk.get("section", "")
        content_type = chunk.get("content_type", "")
        kb_name = chunk.get("kb_name", "")
        
        if not text or len(text) < 50:  # 跳过太短的文本
            continue
        
        print(f"\n[{i+1}/{len(all_chunks)}] 处理: {chunk_id}")
        print(f"  章节: {section} | 类型: {content_type} | 来源: {kb_name}")
        print(f"  文本长度: {len(text)} 字符")
        
        # 生成问题
        question = generate_question(text)
        
        if question:
            training_data.append({
                "query": question,
                "positive": text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "section": section,
                    "content_type": content_type,
                    "kb_name": kb_name
                }
            })
            print(f"  生成问题: {question[:50]}...")
        else:
            failed_count += 1
            print(f"  生成失败")
        
        # 控制 API 调用频率
        time.sleep(0.5)
        
        # 每 50 条保存一次进度
        if (i + 1) % 50 == 0:
            save_path = f"offline/training_data_progress_{i+1}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            print(f"\n已保存进度: {save_path} ({len(training_data)} 条)")
    
    # 保存最终结果
    output_path = "offline/embedding_training_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"训练数据生成完成！")
    print(f"{'='*60}")
    print(f"文档块总数: {len(all_chunks)}")
    print(f"成功生成: {len(training_data)}")
    print(f"失败: {failed_count}")
    print(f"保存路径: {output_path}")
    
    # 同时生成 sentence-transformers 格式
    st_output_path = "offline/embedding_training_st.jsonl"
    with open(st_output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps({
                "sentence1": item["query"],
                "sentence2": item["positive"]
            }, ensure_ascii=False) + "\n")
    
    print(f"Sentence-Transformers 格式: {st_output_path}")

if __name__ == "__main__":
    main()

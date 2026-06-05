"""
生成 BGE-M3 微调训练数据（多线程加速版）
运行：D:\an\envs\project2\python.exe offline/generate_training_data.py
"""
import sys, os, json, time, random
sys.path.insert(0, ".")

from pymilvus import MilvusClient
from app.config import settings
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# DeepSeek API 配置
DEEPSEEK_HOST = settings.DEEPSEEK_HOST
DEEPSEEK_KEY = settings.DEEPSEEK_KEY
DEEPSEEK_MODEL = settings.DEEPSEEK_MODEL

# 并发配置
MAX_WORKERS = 10  # 并发线程数
API_DELAY = 0.1   # 每个线程的请求间隔

# 数据量配置
TARGET_TOTAL = 1200      # 总生成目标（训练+验证）
TRAIN_SIZE = 1000        # 训练集大小
VAL_SIZE = 200           # 验证集大小

# 质量控制配置
MIN_QUESTION_LEN = 10    # 问题最短长度
MAX_QUESTION_LEN = 200   # 问题最长长度（过长可能是乱码或废话）
MIN_CONTENT_LEN = 80     # 文档内容最短长度（提高到80，过滤掉标题和碎片）

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

# 线程安全的计数器和去重集合
lock = threading.Lock()
success_count = 0
fail_count = 0
seen_questions = set()  # 去重用

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
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return ""

def process_chunk(chunk, index, total):
    """处理单个文档块"""
    global success_count, fail_count

    text = chunk.get("text", "")
    chunk_id = chunk.get("chunk_id", f"chunk_{index}")
    kb_name = chunk.get("kb_name", "")
    char_count = chunk.get("char_count", 0)

    # 质量检查1：内容长度
    if not text or len(text) < MIN_CONTENT_LEN:
        return None

    # 质量检查2：内容不能全是数字或符号
    clean_text = text.replace(" ", "").replace("\n", "")
    if len(clean_text) < 30:
        return None

    # 生成问题
    question = generate_question(text)

    time.sleep(API_DELAY)

    with lock:
        # 质量检查3：问题长度
        if question and (len(question) < MIN_QUESTION_LEN or len(question) > MAX_QUESTION_LEN):
            fail_count += 1
            print(f"[{index+1}/{total}] X {chunk_id}: 问题长度异常({len(question)}字符)")
            sys.stdout.flush()
            return None

        # 质量检查4：去重
        if question and question in seen_questions:
            fail_count += 1
            print(f"[{index+1}/{total}] X {chunk_id}: 重复问题")
            sys.stdout.flush()
            return None

        if question:
            seen_questions.add(question)
            success_count += 1
            print(f"[{index+1}/{total}] OK {chunk_id}: {question[:40]}...")
        else:
            fail_count += 1
            print(f"[{index+1}/{total}] X {chunk_id}: 生成失败")
        sys.stdout.flush()

    if question:
        return {
            "query": question,
            "positive": text,
            "metadata": {
                "chunk_id": chunk_id,
                "kb_name": kb_name,
                "char_count": char_count
            }
        }
    return None

def main():
    global success_count, fail_count
    
    # 连接 Milvus
    uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    client = MilvusClient(uri=uri)
    collection = settings.MILVUS_COLLECTION
    
    print(f"连接 Milvus: {uri}")
    print(f"集合: {collection}")
    
    # 用向量搜索获取所有文档块
    print("正在获取所有文档块...")
    
    sample = client.query(
        collection_name=collection,
        filter="",
        output_fields=["embedding"],
        limit=1
    )
    
    if not sample:
        print("错误：Milvus 中没有数据")
        return
    
    query_vector = sample[0]["embedding"]
    
    search_results = client.search(
        collection_name=collection,
        data=[query_vector],
        anns_field="embedding",
        search_params={"metric_type": "IP", "params": {"nprobe": 16}},
        limit=2000,
        output_fields=["chunk_id", "text", "kb_name", "char_count"]
    )
    
    if not search_results or not search_results[0]:
        print("错误：搜索无结果")
        return

    # 去重
    seen_ids = set()
    all_chunks = []
    for hit in search_results[0]:
        entity = hit.get("entity", {})
        chunk_id = entity.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            all_chunks.append(entity)

    print(f"获取到 {len(all_chunks)} 个文档块")
    sys.stdout.flush()
    
    if not all_chunks:
        print("错误：未获取到文档块")
        return
    
    # 检查 API 配置
    if not DEEPSEEK_KEY:
        print("错误：DEEPSEEK_KEY 未配置，请检查 .env 文件")
        return
    
    # 限制生成数量到目标
    if len(all_chunks) > TARGET_TOTAL:
        print(f"文档块数({len(all_chunks)})超过目标({TARGET_TOTAL})，随机采样")
        random.shuffle(all_chunks)
        all_chunks = all_chunks[:TARGET_TOTAL]

    print(f"\n开始生成训练数据...")
    print(f"API: {DEEPSEEK_HOST}")
    print(f"模型: {DEEPSEEK_MODEL}")
    print(f"并发数: {MAX_WORKERS}")
    print(f"目标: 训练集 {TRAIN_SIZE} 条 + 验证集 {VAL_SIZE} 条")
    print(f"预计时间: {len(all_chunks) * 0.5 / MAX_WORKERS:.0f} 秒")
    sys.stdout.flush()

    start_time = time.time()
    
    # 多线程并发处理
    training_data = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, chunk in enumerate(all_chunks):
            future = executor.submit(process_chunk, chunk, i, len(all_chunks))
            futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            if result:
                training_data.append(result)
            
            # 每 100 条保存一次进度
            if len(training_data) % 100 == 0 and len(training_data) > 0:
                save_path = f"offline/training_data_progress_{len(training_data)}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(training_data, f, ensure_ascii=False, indent=2)
                print(f"\n已保存进度: {save_path} ({len(training_data)} 条)")
    
    elapsed = time.time() - start_time

    # 随机打乱并分割训练集/验证集
    random.shuffle(training_data)

    if len(training_data) >= TARGET_TOTAL:
        train_data = training_data[:TRAIN_SIZE]
        val_data = training_data[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    else:
        # 数据不足时按比例分割
        split_idx = int(len(training_data) * TRAIN_SIZE / TARGET_TOTAL)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]

    # 保存训练集
    train_path = "offline/embedding_train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    # 保存验证集
    val_path = "offline/embedding_val.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    # 保存合并版本（兼容旧代码）
    output_path = "offline/embedding_training_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"训练数据生成完成！")
    print(f"{'='*60}")
    print(f"文档块总数: {len(all_chunks)}")
    print(f"成功生成: {success_count}")
    print(f"失败: {fail_count}")
    print(f"实际保存: {len(training_data)}")
    print(f"  - 训练集: {len(train_data)} 条 → {train_path}")
    print(f"  - 验证集: {len(val_data)} 条 → {val_path}")
    print(f"耗时: {elapsed:.1f} 秒")

    # 同时生成 sentence-transformers 格式
    st_train_path = "offline/embedding_train_st.jsonl"
    with open(st_train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps({
                "sentence1": item["query"],
                "sentence2": item["positive"]
            }, ensure_ascii=False) + "\n")

    st_val_path = "offline/embedding_val_st.jsonl"
    with open(st_val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps({
                "sentence1": item["query"],
                "sentence2": item["positive"]
            }, ensure_ascii=False) + "\n")

    print(f"\nSentence-Transformers 格式:")
    print(f"  - 训练集: {st_train_path}")
    print(f"  - 验证集: {st_val_path}")

if __name__ == "__main__":
    main()

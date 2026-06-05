"""
训练数据优化 + 生成训练集/验证集
运行：D:\an\envs\project2\python.exe offline/optimize_training_data.py
"""
import json, re, os, random
from collections import Counter

INPUT_FILE = "offline/embedding_training_data.json"
OUTPUT_DIR = "offline/training_data"

def clean_text(text: str) -> str:
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+-\d+-\d+', '', text)
    text = re.sub(r'(武汉兴图新科电子股份有限公司\s*){2,}', '武汉兴图新科电子股份有限公司 ', text)
    # 清理占位符
    text = text.replace('◆', '').replace('□', '').replace('■', '')
    return text.strip()

def clean_query(query: str) -> str:
    """清理问题"""
    query = query.strip('"\'""\'\'')
    if query.startswith("问题："):
        query = query[3:]
    if query.startswith("Q：") or query.startswith("Q:"):
        query = query[2:]
    return query.strip()

def is_valid_pair(query: str, positive: str) -> bool:
    """检查数据对是否有效"""
    if len(query) < 5 or len(query) > 100:
        return False
    if len(positive) < 50:
        return False
    if any(phrase in query for phrase in ["这段文档", "这段文本", "该文档", "上述内容"]):
        return False
    if not query.endswith("？") and not query.endswith("?"):
        return False
    return True

def get_core_query(query: str) -> str:
    """提取问题核心内容（去除前缀）"""
    core = query
    for prefix in ["请问", "我想知道", "请问一下", "麻烦问一下", "请告诉我"]:
        if core.startswith(prefix):
            core = core[len(prefix):]
    return core

def deduplicate(data: list) -> list:
    """去重：基于问题的核心内容"""
    seen_cores = set()
    unique_data = []
    
    for item in data:
        core = get_core_query(item["query"])
        if core not in seen_cores:
            seen_cores.add(core)
            unique_data.append(item)
    
    return unique_data

def generate_query_variants(query: str) -> list:
    """为一个问题生成多个变体"""
    variants = [query]
    
    # 去除现有前缀
    core = get_core_query(query)
    
    # 生成变体
    if not query.startswith("请问"):
        variants.append(f"请问{core}")
    if not query.startswith("我想知道"):
        variants.append(f"我想知道{core}")
    if not query.startswith("请告诉我"):
        variants.append(f"请告诉我{core}")
    if not query.startswith("麻烦问一下"):
        variants.append(f"麻烦问一下{core}")
    
    return variants

def oversample_with_variants(data: list, target_count: int) -> list:
    """
    通过生成问题变体来增加数据量
    """
    result = []
    idx = 0
    
    while len(result) < target_count:
        item = data[idx % len(data)]
        variants = generate_query_variants(item["query"])
        
        # 选择一个变体
        variant = variants[idx % len(variants)]
        
        # 避免重复
        existing_queries = [x["query"] for x in result]
        if variant not in existing_queries:
            result.append({
                **item,
                "query": variant
            })
        
        idx += 1
        
        # 防止无限循环
        if idx > target_count * 5:
            break
    
    return result[:target_count]

def main():
    random.seed(42)
    
    print("加载训练数据...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"原始数据: {len(raw_data)} 条")
    
    # 清洗
    print("\n清洗数据...")
    cleaned_data = []
    for item in raw_data:
        query = clean_query(item["query"])
        positive = clean_text(item["positive"])
        
        if is_valid_pair(query, positive):
            cleaned_data.append({
                "query": query,
                "positive": positive,
                "metadata": item.get("metadata", {})
            })
    
    print(f"清洗后: {len(cleaned_data)} 条")
    
    # 按知识库分组去重
    print("\n按知识库分组去重...")
    kb_groups = {}
    for item in cleaned_data:
        kb = item.get("metadata", {}).get("kb_name", "unknown")
        if kb not in kb_groups:
            kb_groups[kb] = []
        kb_groups[kb].append(item)
    
    unique_groups = {}
    for kb, items in kb_groups.items():
        unique_groups[kb] = deduplicate(items)
        print(f"  {kb}: {len(items)} → {len(unique_groups[kb])} 条")
    
    # 过采样招股说明书2到目标数量
    target_kb2 = 250  # 目标数量
    print(f"\n过采样招股说明书2到 {target_kb2} 条...")
    kb2_oversampled = oversample_with_variants(unique_groups["招股说明书2"], target_kb2)
    print(f"  招股说明书2: {len(unique_groups['招股说明书2'])} → {len(kb2_oversampled)} 条")
    
    # 合并数据
    all_data = unique_groups["招股说明书1"] + kb2_oversampled
    random.shuffle(all_data)
    
    print(f"\n合并后: {len(all_data)} 条")
    
    # 分割训练集和验证集 (80% / 20%)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存数据
    with open(f"{OUTPUT_DIR}/all_data.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{OUTPUT_DIR}/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{OUTPUT_DIR}/val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open(f"{OUTPUT_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps({
                "sentence1": item["query"],
                "sentence2": item["positive"]
            }, ensure_ascii=False) + "\n")
    
    with open(f"{OUTPUT_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps({
                "sentence1": item["query"],
                "sentence2": item["positive"]
            }, ensure_ascii=False) + "\n")
    
    # 统计信息
    print(f"\n{'='*60}")
    print("数据统计")
    print(f"{'='*60}")
    
    query_lens = [len(item["query"]) for item in all_data]
    print(f"问题平均长度: {sum(query_lens)/len(query_lens):.0f} 字符")
    print(f"问题最短: {min(query_lens)} 字符")
    print(f"问题最长: {max(query_lens)} 字符")
    
    pos_lens = [len(item["positive"]) for item in all_data]
    print(f"\n正文平均长度: {sum(pos_lens)/len(pos_lens):.0f} 字符")
    print(f"正文最短: {min(pos_lens)} 字符")
    print(f"正文最长: {max(pos_lens)} 字符")
    
    kb_counts = Counter(item.get("metadata", {}).get("kb_name", "unknown") for item in all_data)
    print(f"\n知识库来源:")
    for kb, count in kb_counts.most_common():
        print(f"  {kb}: {count} 条 ({count/len(all_data)*100:.1f}%)")
    
    # 检查重复
    query_counts = Counter(item['query'] for item in all_data)
    duplicates = {q: c for q, c in query_counts.items() if c > 1}
    print(f"\n重复问题: {len(duplicates)} 个")
    
    print(f"\n输出目录: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

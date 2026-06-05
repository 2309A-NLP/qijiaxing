"""微调前检查脚本"""
import sys, os, json, torch

print("="*60)
print("微调环境检查")
print("="*60)

# 1. 训练数据
print("\n[1/5] 训练数据检查")
train_path = "offline/embedding_train.json"
val_path = "offline/embedding_val.json"

if os.path.exists(train_path):
    with open(train_path, "r", encoding="utf-8") as f:
        train = json.load(f)
    print(f"  训练集: {len(train)} 条")
    print(f"  字段: {list(train[0].keys())}")
    print(f"  示例query: {train[0]['query'][:50]}...")
else:
    print(f"  错误: {train_path} 不存在")

if os.path.exists(val_path):
    with open(val_path, "r", encoding="utf-8") as f:
        val = json.load(f)
    print(f"  验证集: {len(val)} 条")
else:
    print(f"  错误: {val_path} 不存在")

# 2. 依赖库
print("\n[2/5] 依赖库检查")
deps = ["sentence_transformers", "peft", "torch"]
for dep in deps:
    try:
        mod = __import__(dep)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  {dep}: {ver}")
    except ImportError:
        print(f"  {dep}: 未安装!")

# 3. GPU
print("\n[3/5] GPU检查")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  显存: {mem_gb:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("  无GPU，将使用CPU训练（非常慢）")

# 4. 模型
print("\n[4/5] 模型检查")
from sentence_transformers import SentenceTransformer
model_name = "BAAI/bge-base-en-v1.5"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_cache = os.path.join(cache_dir, "models--BAAI--bge-base-en-v1.5")
if os.path.exists(model_cache):
    print(f"  {model_name}: 已缓存")
else:
    print(f"  {model_name}: 需要下载（约400MB）")

# 5. LoRA配置
print("\n[5/5] LoRA配置预览")
print(f"  r=16, alpha=32")
print(f"  target_modules: query, key, value")
print(f"  dropout=0.1")
print(f"  预计可训练参数: ~0.5M (总参数的0.5%)")

# 计算训练步数
if os.path.exists(train_path):
    batch_size = 8
    epochs = 3
    steps_per_epoch = len(train) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * 0.1)
    print(f"\n训练步数:")
    print(f"  每epoch: {steps_per_epoch} 步")
    print(f"  总步数: {total_steps} 步")
    print(f"  warmup: {warmup_steps} 步")
    print(f"  预计耗时: {total_steps * 0.5 / 60:.1f} 分钟 (GPU)")

print("\n" + "="*60)
print("检查完成")
print("="*60)

"""加载缓存模型并运行微调"""
import os, sys

# 强制离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
# 清除可能错误的 HF_ENDPOINT
os.environ.pop('HF_ENDPOINT', None)

# 明确设置缓存路径
cache_dir = os.path.join(os.environ.get('USERPROFILE', 'C:\\Users\\qjx'), '.cache', 'huggingface', 'hub')
os.environ['HF_HOME'] = cache_dir

# 现在导入
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses, evaluation
from torch.utils.data import DataLoader
import json
import logging
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("finetune_embedding.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)

# 1. 检查缓存
model_id = "BAAI/bge-base-zh-v1.5"
cache_path = os.path.join(cache_dir, "models--BAAI--bge-base-zh-v1.5", "snapshots")
if os.path.exists(cache_path):
    snaps = os.listdir(cache_path)
    if snaps:
        model_path = os.path.join(cache_path, snaps[-1])
        logger.info(f"缓存模型路径: {model_path}")
        logger.info(f"文件: {os.listdir(model_path)}")

# 2. 加载数据
logger.info("加载训练数据...")
with open("training_data/train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open("training_data/val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
train_examples = [InputExample(texts=[QUERY_PREFIX + item["query"], item["positive"]]) for item in train_data]
val_examples = [InputExample(texts=[QUERY_PREFIX + item["query"], item["positive"]]) for item in val_data]

logger.info(f"训练样本: {len(train_examples)}, 验证样本: {len(val_examples)}")

# 3. 加载模型（离线模式）
logger.info(f"加载模型: {model_id}")
model = SentenceTransformer(model_id, cache_folder=cache_dir)
logger.info("模型加载成功!")

# 4. 应用 LoRA
logger.info("应用 LoRA...")
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1, bias="none", task_type="FEATURE_EXTRACTION",
)
if hasattr(model[0], 'auto_model'):
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
trainable = sum(p.numel() for p in model[0].auto_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model[0].auto_model.parameters())
logger.info(f"可训练: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# 5. 创建评估器
import random
random.seed(42)
val_s1, val_s2, val_sc = [], [], []
for ex in val_examples:
    val_s1.append(ex.texts[0]); val_s2.append(ex.texts[1]); val_sc.append(1.0)
all_pos = [ex.texts[1] for ex in val_examples]
for i, ex in enumerate(val_examples):
    neg = random.choice([j for j in range(len(all_pos)) if j != i])
    val_s1.append(ex.texts[0]); val_s2.append(all_pos[neg]); val_sc.append(0.0)
evaluator = evaluation.EmbeddingSimilarityEvaluator(val_s1, val_s2, val_sc, name="validation", show_progress_bar=True, batch_size=16)

# 6. 设置训练
train_dl = DataLoader(train_examples, batch_size=16, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

output_path = "finetuned-bge-base-zh-v1.5-lora"
os.makedirs(output_path, exist_ok=True)

peft_ref = model[0].auto_model

logger.info("开始训练...")
model.fit(
    train_objectives=[(train_dl, train_loss)],
    epochs=3, warmup_steps=100,
    evaluator=evaluator, output_path=output_path,
    save_best_model=True,
    optimizer_params={"lr": 2e-5},
    show_progress_bar=True,
)

# 7. 合并 LoRA
logger.info("合并 LoRA 权重...")
try:
    merged = peft_ref.merge_and_unload()
    model[0].auto_model = merged
    model.save(output_path)
    logger.info(f"合并完成: {output_path}")
except Exception as e:
    logger.warning(f"合并失败: {e}")

# 8. 最终评估
logger.info("最终评估...")
metrics = evaluator(model)
logger.info(f"结果: {metrics}")
logger.info("=" * 50)
logger.info("训练完成!")
logger.info(f"模型路径: {output_path}")
logger.info(f"评估: {metrics}")
logger.info("=" * 50)

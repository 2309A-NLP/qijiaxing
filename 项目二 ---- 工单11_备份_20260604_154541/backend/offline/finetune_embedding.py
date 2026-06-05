#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BGE-Base-ZH-v1.5 微调训练脚本

功能：
1. 加载训练数据（JSON 格式）
2. 训练 Embedding 模型（Sentence-Transformers）
3. 支持全参数微调和 LoRA 微调
4. 自动评估并保存最优模型

使用方法：
    python finetune_embedding.py --model BAAI/bge-base-zh-v1.5 --mode full
    python finetune_embedding.py --model BAAI/bge-base-zh-v1.5 --mode lora

作者：尤明曦
日期：2026-06-02
"""

import os
# 强制离线模式，阻止任何网络连接
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HOME'] = os.path.join(
    os.environ.get('USERPROFILE', 'C:\\Users\\qjx'),
    '.cache', 'huggingface'
)

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from sentence_transformers import evaluation
from peft import LoraConfig, get_peft_model, PeftModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("finetune_embedding.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_training_data(
    train_path: str, val_path: str
) -> tuple[List[InputExample], List[InputExample]]:
    """
    加载训练数据并转换为 Sentence-Transformers 格式

    Args:
        train_path: 训练数据路径（JSON 格式）
        val_path: 验证数据路径（JSON 格式）

    Returns:
        train_examples: 训练样本列表
        val_examples: 验证样本列表

    数据格式：
        [
            {
                "query": "用户问题",
                "positive": "相关文档片段",
                "metadata": {"chunk_id": "...", "kb_name": "...", "char_count": 800}
            },
            ...
        ]
    """
    logger.info(f"加载训练数据: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    logger.info(f"加载验证数据: {val_path}")
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # 转换为 InputExample 格式
    # 注意：BGE 模型对 query 有特殊 instruction 前缀
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    train_examples = [
        InputExample(texts=[QUERY_PREFIX + item["query"], item["positive"]])
        for item in train_data
    ]

    val_examples = [
        InputExample(texts=[QUERY_PREFIX + item["query"], item["positive"]])
        for item in val_data
    ]

    logger.info(f"训练样本数: {len(train_examples)}")
    logger.info(f"验证样本数: {len(val_examples)}")

    return train_examples, val_examples


def create_evaluator(val_examples: List[InputExample]) -> evaluation.EmbeddingSimilarityEvaluator:
    """
    创建验证集评估器

    Args:
        val_examples: 验证样本列表

    Returns:
        evaluator: EmbeddingSimilarityEvaluator 实例

    说明：
        - 用于每个 epoch 结束后评估模型效果
        - 配合 save_best_model=True 使用，保存最优模型
        - 评估指标：Spearman 相关系数
        - 包含正例（score=1.0）和随机负例（score=0.0），确保相关系数可计算
    """
    import random
    random.seed(42)

    val_sentences1 = []
    val_sentences2 = []
    val_scores = []

    # 正例：query 和对应 positive（score=1.0）
    for ex in val_examples:
        val_sentences1.append(ex.texts[0])
        val_sentences2.append(ex.texts[1])
        val_scores.append(1.0)

    # 负例：query 和随机不匹配的 positive（score=0.0）
    all_positives = [ex.texts[1] for ex in val_examples]
    for i, ex in enumerate(val_examples):
        # 随机选一个不匹配的 positive
        neg_idx = random.choice([j for j in range(len(all_positives)) if j != i])
        val_sentences1.append(ex.texts[0])
        val_sentences2.append(all_positives[neg_idx])
        val_scores.append(0.0)

    logger.info(f"验证集: {len(val_examples)} 正例 + {len(val_examples)} 负例 = {len(val_scores)} 对")

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        val_sentences1,
        val_sentences2,
        val_scores,
        name="validation",
        show_progress_bar=True,
        batch_size=16,
    )

    return evaluator


def setup_lora_model(model: SentenceTransformer) -> SentenceTransformer:
    """
    配置 LoRA 微调

    Args:
        model: 原始 SentenceTransformer 模型

    Returns:
        model: 应用 LoRA 后的模型

    LoRA 配置说明：
        - r=8: 低秩，控制参数量（越小参数越少，但可能损失信息）
        - lora_alpha=16: 缩放系数（通常设为 2*r）
        - target_modules: 应用 LoRA 的层（query, key, value 注意力层）
        - lora_dropout=0.1: 防止过拟合

    显存优化：
        - 全参数微调需要 ~7-8GB
        - LoRA 微调只需 ~5-6GB
        - 8GB VRAM 推荐用 LoRA
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    # 获取底层 Transformer 模型并应用 LoRA
    if hasattr(model[0], 'auto_model'):
        import torch.nn as _nn
        # auto_model 是返回 self.model 的方法，直接操作 model 属性
        _transformer = model[0]
        if isinstance(_transformer, _nn.Module):
            _transformer.model = get_peft_model(_transformer.model, lora_config)
        else:
            model[0].model = get_peft_model(model[0].model, lora_config)
    else:
        logger.error("无法获取底层 Transformer 模型，请检查模型结构")
        sys.exit(1)

    # 打印可训练参数量
    trainable_params = sum(
        p.numel() for p in model[0].auto_model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model[0].auto_model.parameters())
    logger.info(f"LoRA 可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def train_model(
    model_name: str,
    train_examples: List[InputExample],
    val_examples: List[InputExample],
    output_dir: str,
    mode: str = "full",
    epochs: int = 3,
    batch_size: int = 8,
    warmup_steps: int = 100,
    learning_rate: float = 2e-5,
) -> str:
    """
    训练 Embedding 模型

    Args:
        model_name: 基础模型名称（如 "BAAI/bge-large-zh-v1.5"）
        train_examples: 训练样本列表
        val_examples: 验证样本列表
        output_dir: 输出目录
        mode: 训练模式（"full" 全参数微调 或 "lora" LoRA 微调）
        epochs: 训练轮数
        batch_size: 批次大小
        warmup_steps: 预热步数
        learning_rate: 学习率

    Returns:
        output_path: 最优模型保存路径

    训练策略：
        1. 使用 MultipleNegativesRankingLoss（对比学习）
        2. batch 内其他样本自动作为负样本
        3. 每个 epoch 结束后在验证集上评估
        4. 保存验证分数最高的模型

    显存优化：
        - batch_size=8: ~7-8GB（全参数微调）
        - batch_size=16: ~5-6GB（LoRA 微调）
        - OOM 时减小 batch_size 或使用 gradient checkpointing
    """
    logger.info(f"加载基础模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 应用 LoRA（如果指定）
    if mode == "lora":
        logger.info("应用 LoRA 微调")
        model = setup_lora_model(model)

    # 创建数据加载器
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    # 损失函数：对比学习
    # MultipleNegativesRankingLoss 自动将 batch 内其他样本作为负样本
    # batch_size 越大，隐式负样本越多，效果越好
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 创建评估器
    evaluator = create_evaluator(val_examples)

    # 输出目录
    output_path = os.path.join(output_dir, f"finetuned-{model_name.split('/')[-1]}-{mode}")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    logger.info(f"开始训练: mode={mode}, epochs={epochs}, batch_size={batch_size}")

    # LoRA 模式：创建 evaluator 包装器，在训练过程中保存 LoRA 权重
    if mode == "lora":
        adapter_dir = output_path + "_adapter"
        os.makedirs(adapter_dir, exist_ok=True)
        # 创建 adapter_config.json（手动保存，因为 save_pretrained 在 ST 模型中会保存完整模型）
        import json as _json
        _adapter_config = {
            "peft_type": "LORA",
            "task_type": "FEATURE_EXTRACTION",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["query", "key", "value"],
            "lora_dropout": 0.1,
            "bias": "none",
            "base_model_name_or_path": model_name,
        }
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            _json.dump(_adapter_config, f, indent=2)
        # 保存初始 LoRA state dict
        _lora_keys = {k: v for k, v in model[0].auto_model.state_dict().items() if "lora" in k}
        torch.save(_lora_keys, os.path.join(adapter_dir, "adapter_model.bin"))
        logger.info(f"LoRA adapter 初始备份至: {adapter_dir}")

        # 包装 evaluator：每次评估后保存当前 LoRA 权重
        original_evaluator = evaluator
        class _SaveLoraEval:
            def __init__(self, base_eval, save_dir, model_ref):
                self.base_eval = base_eval
                self.save_dir = save_dir
                self.model_ref = model_ref
            def __call__(self, model, *args, **kwargs):
                result = self.base_eval(model, *args, **kwargs)
                try:
                    # 仅保存 LoRA 权重（不保存完整模型）
                    trans = self.model_ref[0]
                    model_obj = getattr(trans, 'model', None) or getattr(trans, 'auto_model', lambda: None)()
                    if hasattr(model_obj, 'state_dict'):
                        lora_st = {k: v for k, v in model_obj.state_dict().items() if "lora" in k}
                        if lora_st:
                            torch.save(lora_st, os.path.join(self.save_dir, "adapter_model.bin"))
                            logger.info(f"训练中 LoRA 权重已更新至: {self.save_dir}")
                except Exception as e:
                    logger.warning(f"保存训练中 LoRA 权重失败: {e}")
                return result
        evaluator = _SaveLoraEval(original_evaluator, adapter_dir, model)

    # 训练（不传 output_path，防止 fit() 自动保存重载破坏 PeftModel）
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        save_best_model=False,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    # LoRA 模式：合并权重后再保存
    if mode == "lora":
        logger.info("合并 LoRA 权重到基础模型...")
        logger.info(f"model[0].model type: {type(model[0].model).__name__}")
        if hasattr(model[0].model, 'merge_and_unload'):
            merged_model = model[0].model.merge_and_unload()
            model[0].model = merged_model
            model.save(output_path)
            logger.info(f"LoRA 权重合并完成，已保存至: {output_path}")
        else:
            # fallback: 用 evaluator 最后保存的 LoRA 权重重建
            logger.warning("PeftModel 被破坏，从备份 adapter 重建...")
            try:
                base = SentenceTransformer(model_name)
                lora_cfg = LoraConfig(
                    r=8, lora_alpha=16,
                    target_modules=["query", "key", "value"],
                    lora_dropout=0.1, bias="none", task_type="FEATURE_EXTRACTION",
                )
                # 应用 LoRA
                logger.info(f"base[0].auto_model type BEFORE: {type(base[0].auto_model).__name__}")
                # 直接通过 _modules 字典操作，绕过可能的 __getitem__ 副本问题
                first_key = list(base._modules.keys())[0]
                transformer = base._modules[first_key]
                logger.info(f"transformer type: {type(transformer).__name__}, key: {first_key}")
                logger.info(f"transformer.auto_model type: {type(transformer.auto_model).__name__}")
                # auto_model 是返回 self.model 的方法，直接操作 model 属性
                peft_model = get_peft_model(transformer.model, lora_cfg)
                logger.info(f"peft_model type: {type(peft_model).__name__}")
                logger.info(f"has merge_and_unload: {hasattr(peft_model, 'merge_and_unload')}")
                transformer.model = peft_model
                logger.info(f"transformer.model type: {type(transformer.model).__name__}")
                # 加载训练好的 LoRA 权重
                ap = os.path.join(adapter_dir, "adapter_model.bin")
                if os.path.exists(ap):
                    st = torch.load(ap, map_location="cuda", weights_only=True)
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(transformer.model, st, adapter_name="default")
                # 验证是否为 PeftModel
                if hasattr(transformer.model, 'merge_and_unload'):
                    merged = transformer.model.merge_and_unload()
                    transformer.model = merged
                    base.save(output_path)
                    model = base
                    logger.info(f"LoRA 合并完成，已保存至: {output_path}")
                else:
                    raise RuntimeError("get_peft_model 返回的不是 PeftModel")
            except Exception as e2:
                logger.error(f"所有合并方案失败: {e2}")
                model.save(output_path)

    logger.info(f"训练完成，最优模型保存至: {output_path}")

    return output_path


def evaluate_model(model_path: str, val_examples: List[InputExample]) -> Dict[str, float]:
    """
    评估微调后的模型

    Args:
        model_path: 模型路径
        val_examples: 验证样本列表

    Returns:
        metrics: 评估指标字典

    评估指标：
        - spearman_correlation: Spearman 相关系数（主要指标）
        - pearson_correlation: Pearson 相关系数
    """
    logger.info(f"加载模型进行评估: {model_path}")
    model = SentenceTransformer(model_path)

    evaluator = create_evaluator(val_examples)
    metrics = evaluator(model)

    logger.info(f"评估结果: {metrics}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="BGE Embedding 模型微调训练脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-base-zh-v1.5",
        help="基础模型名称（默认: BAAI/bge-base-zh-v1.5）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "lora"],
        default="lora",
        help="训练模式: full（全参数微调）或 lora（LoRA 微调）（默认: lora）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数（默认: 3）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小（默认: 8，LoRA 可设为 16）",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="预热步数（默认: 100）",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="学习率（默认: 2e-5）",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="training_data/train.json",
        help="训练数据路径（默认: training_data/train.json）",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="training_data/val.json",
        help="验证数据路径（默认: training_data/val.json）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="输出目录（默认: 当前目录）",
    )

    args = parser.parse_args()

    # 检查 CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA 不可用，请检查 GPU 驱动和 PyTorch 安装")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 加载数据
    train_examples, val_examples = load_training_data(args.train_data, args.val_data)

    # 训练
    output_path = train_model(
        model_name=args.model,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=args.output_dir,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
    )

    # 评估
    metrics = evaluate_model(output_path, val_examples)

    logger.info("=" * 50)
    logger.info("训练完成！")
    logger.info(f"最优模型路径: {output_path}")
    logger.info(f"评估指标: {metrics}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

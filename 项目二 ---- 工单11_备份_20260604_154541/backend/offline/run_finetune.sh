#!/bin/bash
# BGE Embedding 模型微调启动脚本
# 使用方法：
#   ./run_finetune.sh full   # 全参数微调
#   ./run_finetune.sh lora   # LoRA 微调（默认）

MODE=${1:-lora}

echo "=========================================="
echo "BGE-Base-EN-v1.5 微调训练"
echo "=========================================="
echo "模式: $MODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '未检测到')"
echo "显存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo '未知')"
echo "=========================================="

if [ "$MODE" = "full" ]; then
    echo "全参数微调模式"
    echo "预计显存占用: ~7-8GB"
    echo "建议 batch_size: 8"
    echo ""
    python finetune_embedding.py \
        --model BAAI/bge-base-zh-v1.5 \
        --mode full \
        --epochs 3 \
        --batch_size 8 \
        --warmup_steps 100 \
        --learning_rate 2e-5 \
        --train_data training_data/train.json \
        --val_data training_data/val.json \
        --output_dir .
elif [ "$MODE" = "lora" ]; then
    echo "LoRA 微调模式"
    echo "预计显存占用: ~5-6GB"
    echo "建议 batch_size: 16"
    echo ""
    python finetune_embedding.py \
        --model BAAI/bge-base-zh-v1.5 \
        --mode lora \
        --epochs 3 \
        --batch_size 16 \
        --warmup_steps 100 \
        --learning_rate 2e-5 \
        --train_data training_data/train.json \
        --val_data training_data/val.json \
        --output_dir .
else
    echo "未知模式: $MODE"
    echo "使用方法: ./run_finetune.sh [full|lora]"
    exit 1
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

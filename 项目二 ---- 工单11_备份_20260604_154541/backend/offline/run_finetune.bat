@echo off
set TRANSFORMERS_OFFLINE=1
set HF_HUB_OFFLINE=1
set HF_DATASETS_OFFLINE=1
cd /d C:\Users\qjx\Desktop\github\项目二 ---- 工单2\backend\offline
if exist finetuned-bge-base-zh-v1.5-lora rmdir /s /q finetuned-bge-base-zh-v1.5-lora
D:\an\envs\project2\python.exe finetune_embedding.py --model BAAI/bge-base-zh-v1.5 --mode lora --epochs 3 --batch_size 16 --warmup_steps 100 --learning_rate 2e-5 --train_data training_data/train.json --val_data training_data/val.json --output_dir .

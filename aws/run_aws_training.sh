#!/usr/bin/env bash
set -e

echo "============================================================"
echo "🚀 VIRTUS-ML: AWS GPU QWEN 2.5 7B QLoRA TRAINING RUNNER"
echo "============================================================"

# 1. Check GPU
echo "Checking GPU availability..."
nvidia-smi

# 2. Install modern training dependencies
echo "Installing GPU dependencies (PyTorch, PEFT, BitsAndBytes, TRL)..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers peft bitsandbytes trl datasets accelerate scipy

# 3. Run Qwen 2.5 7B QLoRA Training
echo "Starting Qwen 2.5 7B QLoRA Training..."
python3 aws/qwen_finetune.py \
    --data aws/data/train_pairs_sample.jsonl \
    --output aws/models/qwen_er_adapter \
    --epochs 2 \
    --batch-size 4

echo "============================================================"
echo "✅ TRAINING COMPLETE!"
echo "Trained adapter saved to: aws/models/qwen_er_adapter (~100 MB)"
echo "⚠️ REMINDER: Stop this EC2 instance to prevent idle charges!"
echo "============================================================"

#!/bin/bash

echo "=== Direct 1-GPU Training Test with SmolVLA (no accelerate) ==="
echo "Environment: lerobot"
echo "GPU: 1"
echo "Steps: 50 (quick local test)"
echo ""

# Activate conda environment
# TODO should change to your conda environment
source /home/superfun77/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Set CUDA environment for 1 GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=OFF
export CUDA_LAUNCH_BLOCKING=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export HF_ENDPOINT=https://hf-mirror.com


# Change to working directory
cd /opt/projects/news/lerobot

# Set output directory with timestamp
export OUTPUT_DIR="outputs/test_direct_1gpu_local_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Test direct training with 1 GPU (no accelerate)
python -m lerobot.scripts.accelerate_train \
  --policy.type=smolvla2 \
  --policy.load_vlm_weights True \
  --batch_size=16 \
  --policy.push_to_hub=false \
  --dataset.repo_id=aopolin-lv/libero_goal_no_noops_lerobot_v21 \
  --dataset.video_backend=pyav \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --steps=50 \
  --save_freq=25 \
  --num_workers=0 \
  --output_dir=$OUTPUT_DIR \
  --job_name=libero_smolvla2_acc_test \
  --wandb.enable=true

echo ""
echo "=== Training completed! ==="
echo "Check outputs in: $OUTPUT_DIR"
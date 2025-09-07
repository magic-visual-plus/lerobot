#!/bin/bash
# Create logs directory if it doesn't exist
mkdir -p /opt/product/lerobot/logs

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 2-GPU Test CUDA environment with improved stability settings
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_DEEPSPEED=false
export HF_ACCELERATE_DEVICE_MAP=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export SAFETENSORS_FAST_GPU=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export ACCELERATE_TORCH_DEVICE_MAP_AUTO=false
# Additional settings to help with multi-GPU stability
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TREE_THRESHOLD=0
export OMP_NUM_THREADS=1
# Force PyTorch to use consistent device placement
export TORCH_CUDA_ARCH_LIST="8.0"

# Change to working directory
cd /opt/product/lerobot

echo "=== Testing Fixed Accelerate Multi-GPU Training with SmolVLA2 ==="
echo "Dataset: aopolin-lv/libero_goal_no_noops_lerobot_v21"
echo "GPUs: 2"
echo "Steps: 100 (for quick test)"
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Check GPU availability
nvidia-smi
echo ""

# Set output directory with job ID
export OUTPUT_DIR="/root/autodl-fs/ckpts/test_accelerate_2gpu_fixed_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Test accelerate training with reduced batch size for stability
accelerate launch --config_file accelerate_configs/2gpu_config_safe.yaml -m lerobot.scripts.accelerate_train \
    --policy.type=smolvla2 \
    --policy.load_vlm_weights True \
    --batch_size=48 \
    --policy.push_to_hub=false \
    --dataset.repo_id=aopolin-lv/libero_goal_no_noops_lerobot_v21 \
    --dataset.video_backend=pyav \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --steps=100000 \
    --save_freq=10000 \
    --output_dir=$OUTPUT_DIR \
    --job_name=libero_goal_0907_smolvla2_2gpu_fixed_test \
    --wandb.enable=true

echo ""
if [ $? -eq 0 ]; then
    echo "=== Training completed successfully! ==="
else
    echo "=== Training failed with exit code $? ==="
fi
echo "Check logs and outputs in: $OUTPUT_DIR"
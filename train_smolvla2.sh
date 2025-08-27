rm -rf outputs/train/libero_smolvla2_scratch
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" python -m lerobot.scripts.train \
  --policy.type=smolvla2 \
  --policy.load_vlm_weights True \
  --dataset.repo_id=aopolin-lv/libero_spatial_no_noops_lerobot_v21 \
  --batch_size=64 \
  --steps=100000 \
  --policy.freeze_vision_encoder=false  --policy.train_expert_only=false \
  --wandb.enable=true \
  --save_freq 10000 \
  --output_dir=outputs/train/libero_smolvla2_scratch \
  --job_name=libero_smolvla2_scratch \
  --policy.push_to_hub=False

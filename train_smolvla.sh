rm -rf outputs/train/libero_smolvla_scratch
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" proxychains python -m lerobot.scripts.train \
  --policy.type=smolvla \
  --policy.load_vlm_weights True \
  --dataset.repo_id=aopolin-lv/libero_spatial_no_noops_lerobot_v21 \
  --batch_size=64 \
  --steps=200000 \
  --wandb.enable=false \
  --save_freq 10000 \
  --output_dir=outputs/train/libero_smolvla_scratch \
  --job_name=libero_smolvla_scratch \
  --policy.push_to_hub=False
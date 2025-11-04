rm -rf outputs/train/1104/goal_single_task_5_ds_disable_lang
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" python -m lerobot.scripts.train \
  --policy.type=smolvla4 \
  --policy.load_vlm_weights True \
  --dataset.repo_id=/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds_origin \
  --batch_size=16 \
  --steps=100000 \
  --policy.freeze_vision_encoder=false  --policy.train_expert_only=false \
  --wandb.enable=true \
  --save_freq 5000 \
  --output_dir=outputs/train/1104/goal_single_task_5_ds_disable_lang \
  --job_name=goal_single_task_5_ds_disable_lang \
  --policy.push_to_hub=False

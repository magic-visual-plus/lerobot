rm -rf outputs/train/1104/goal_single_task_5_ds_disable_lang_with_pre
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" python -m lerobot.scripts.train \
  --policy.path=/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1104/libero_smolvla4_1030_goal_autodl_bbox_pretrain/pretrained_model_2w \
  --dataset.repo_id=/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds_origin \
  --batch_size=16 \
  --steps=100000 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=false \
  --wandb.enable=true \
  --save_freq 10000 \
  --output_dir=outputs/train/1104/goal_single_task_5_ds_disable_lang_with_pre \
  --job_name=goal_single_task_5_ds_disable_lang_with_pre \
  --policy.push_to_hub=false

rm -rf /root/autodl-fs/ckpts/1105/libero_smolvla4_1105_goal_keypint3d_with_pretrain
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES="0" python -m lerobot.scripts.train \
  --policy.path=/root/autodl-fs/ckpts/1105/libero_smolvla4_1105_new_goal_pre_train_keypoint3d/checkpoints/010000_bak/pretrained_model \
  --dataset.repo_id=/autodl-fs/data/datasets/libero_goal_no_lerobot_with_com2 \
  --batch_size=40 \
  --steps=100000 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=false \
  --wandb.enable=true \
  --save_freq 5000 \
  --output_dir=/root/autodl-fs/ckpts/1105/libero_smolvla4_1105_goal_keypint3d_with_pretrain \
  --job_name=libero_smolvla4_1105_goal_keypint3d_with_pretrain \
  --policy.push_to_hub=false

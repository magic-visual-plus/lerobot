# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol.py --policy_path=/opt/projects/ckpts/libero_smovla_scrath/040000/pretrained_model

# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol.py --policy_path=lerobot/smolvla_base --video_out_path=data/libero/videos_base_test

# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol2.py --policy_path=/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0828/checkpoints/100000/pretrained_model

# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol2.py --policy_path=/opt/projects/news/lerobot/outputs/train/autodl-step1/060000/pretrained_model

# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol2.py --policy_path=/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0829_goal/checkpoints/100000/pretrained_model --video_out_path=data/libero/videos_goal_0830_10_test --task_suite_name=libero_goal

# echo "next"
# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol2.py --policy_path=/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0829_goal/checkpoints/050000/pretrained_model --video_out_path=data/libero/videos_goal_0830_05_test --task_suite_name=libero_goal

# echo "next"
proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol2.py --policy_path=/opt/projects/news/lerobot/ckpts/autodl/one/040000/pretrained_model --video_out_path=data/libero/videos_goal_0906_one --task_suite_name=libero_goal

# test_6d
# proxychains python /opt/projects/news/lerobot/examples/libero/eval_smol26d.py --policy_path=/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0829_goal_6d/checkpoints/020000/pretrained_model --video_out_path=data/libero/videos_goal6d_0904_06_test --task_suite_name=libero_goal

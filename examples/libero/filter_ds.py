from typing import Any
from pprint import pprint
from numpy.strings import startswith
from loguru import logger
import torch
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)
import cv2
import shutil
from torchvision import transforms

from eval_save import LIBERO_FEATURES

"""
過濾原來的數據集，抽取特定的數據
"""

to_pil = transforms.ToPILImage()

# dataset_path = "/opt/projects/news/lerobot/data/10.13/merged_goal_eval_add3"
dataset_path = "/opt/projects/news/lerobot/data/10.13/libero_goal_no_lerobot_0"
# new_ds_path = "/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds"
new_ds_path = "/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds_origin"

logger.info("clean new ds {}", new_ds_path)
shutil.rmtree(new_ds_path, ignore_errors=True)

new_dataset = LeRobotDataset.create(
    repo_id="new_goal_bbox",
    root=new_ds_path,
    fps=20,
    robot_type="franka",
    features=LIBERO_FEATURES,
)

def convert_tensor_to_numpy(image_tensor):
    pil_img: Any = to_pil(image_tensor)
    cv_img = np.asarray(pil_img)
    # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)  # 转成 BGR
    return cv_img


dataset = LeRobotDataset(dataset_path)
total_episodes = dataset.num_episodes

remove_feature = [
    "index",
    "task",
    "episode_index",
    "task_index",
    "timestamp",
    "frame_index",
    # 'observation.states.gripper_state',
    # 'observation.images.wrist_depth',
    # 'observation.images.segmentation',
    # 'observation.states.ee_state',
    # 'observation.states.joint_state',
    # 'observation.images.depth',
    # 'observation.images.wrist_segmentation'
]

need_tasks = [
    'push the plate to the front of the stove'
]

# 单行
for episode_idx in range(total_episodes):
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    task_desc = dataset.meta.episodes[episode_idx]["tasks"][0]
    print(
        f"Episode {episode_idx}, tasks: {task_desc}, start idx {from_idx}, end idx {to_idx}"
    )
    if task_desc not in need_tasks:
        continue
        
    # if episode_idx in need_clean_eposide:
    #     continue
    # read episode all frames
    episode_frames = []
    
    for i in np.arange(from_idx, to_idx).tolist():
        frame = dataset[i]
        for key in frame.keys():
            if key.startswith("observation.images"):
                frame[key] = convert_tensor_to_numpy(frame[key])
        for feat in remove_feature:
            if feat in frame:
                del frame[feat]
        new_dataset.add_frame(frame, task_desc)
        episode_frames.append(frame)
    new_dataset.save_episode()
    # if episode_idx > 10:
        # break

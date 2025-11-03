from typing import Any
from pprint import pprint
import torch
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from loguru import logger
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import cv2
from torchvision import transforms

# 分析训练的数据分布，单个任务做了什么逻辑， 目前看初始化的state会有微小的变换， 场景的state也不一样

to_pil = transforms.ToPILImage()

dataset_path = '/opt/projects/openpi/datasets/aopolin-lv/libero_goal_no_noops_lerobot_v21'
dataset_path = "/opt/projects/news/lerobot/data/10.13/libero_goal_single_task5_with_com1"

dataset = LeRobotDataset(dataset_path)
ds_meta = LeRobotDatasetMetadata(dataset_path)

print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

image_out_dir = "/opt/projects/news/lerobot/data/debug/goal"

total_episodes = dataset.num_episodes
select_episode_idx = []
for episode_idx in range(total_episodes):
    episode = dataset[episode_idx]
    tasks = dataset.meta.episodes[episode_idx]["tasks"][0]
    # print(f"{dataset[episode_idx]['observation.state'].shape=}")  # (6, c)
    # print()
    from_idx = dataset.episode_data_index["from"][episode_idx].item()
    to_idx = dataset.episode_data_index["to"][episode_idx].item()
    task_name = 'push the plate to the front of the stove'
    task_name = 'open the top drawer and put the bowl inside'
    if tasks == task_name:
        # print(f"Episode {episode_idx}, tasks: {tasks}")
        # print(f'Episode {episode_idx}, frame start {from_idx}, frame end {to_idx}')
        print(f"Episode {episode_idx}, total frame {to_idx - from_idx} start state {dataset[from_idx]['observation.state'].numpy()}")
        image_tensor = dataset[from_idx]["observation.images.image"]
        # tensor to pil
        pil_img: Any = to_pil(image_tensor)
        image_array = np.asarray(pil_img)
        cv_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)  # 转成 BGR
        # print(f'view image {np.asarray(pil_img)}')
        file_name = f'{image_out_dir}/{episode_idx}_start.png'
        cv2.imwrite(file_name, cv_img)
        select_episode_idx.append(episode_idx)
        

print(f'select_episode_idx {select_episode_idx}')
        
        

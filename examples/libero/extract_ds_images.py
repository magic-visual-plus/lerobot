import dataclasses
from typing import Any
from pprint import pprint
import lerobot
from loguru import logger
import numpy as np
import cv2
import draccus
import imageio
import sys
import torch
import pathlib
import os
import ipdb
np.set_printoptions(precision=5, suppress=True)

from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


seed = 7 
torch.manual_seed(seed)
np.random.seed(seed)
# --- Load Policy ---
to_pil = transforms.ToPILImage()

dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1021/new_goal_autodl_with_bbox_action_randpos/libero_eval_20251022_002541'

dataset = LeRobotDataset(dataset_path)
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

episode_idx = 3
tasks = dataset.meta.episodes[episode_idx]["tasks"][0]

from_idx = dataset.episode_data_index["from"][episode_idx].item()
to_idx = dataset.episode_data_index["to"][episode_idx].item()
print(f"episode idx {episode_idx}, task {tasks}, from idx {from_idx} to idx {to_idx}")

des_dir = "/tmp/aa"

for frame_index in np.arange(from_idx, to_idx).tolist():
    frame = dataset[frame_index]
    agentview_image_bbox = to_pil(frame['observation.images.image'])
    agentview_image_bbox = np.asarray(agentview_image_bbox).copy()
    
    # convert bgr to rgb
    agentview_image_bbox = cv2.cvtColor(agentview_image_bbox, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(os.path.join(des_dir, f"{frame_index}.png"), agentview_image_bbox)
    
    if frame_index == 910:
        print(f"frame index {frame_index}")
        state = frame['observation.state']
        wrist_image = to_pil(frame['observation.images.wrist_image'])
        wrist_image = np.asarray(wrist_image).copy()
        wrist_image = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(des_dir, f"{frame_index}_wrist.jpg"), wrist_image)
        cv2.imwrite(os.path.join(des_dir, f"{frame_index}_main.jpg"), agentview_image_bbox)
        
        print(f'state {state.numpy()}')

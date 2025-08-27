
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com/'
from pathlib import Path
import imageio
import torch
import numpy as np
from torchvision.transforms import ToPILImage

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


dataset_name = "lerobot/pusht"
dataset_name = "aopolin-lv/libero_spatial_no_noops_lerobot_v21"

to_pil = ToPILImage()

# In this case with the standard configuration for Diffusion Policy, it is equivalent to this:
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}

# dataset = LeRobotDataset(dataset_name, root="/opt/projects/openpi/datasets/lerobot", delta_timestamps=delta_timestamps)

dataset = LeRobotDataset(repo_id=dataset_name,  episodes=[0])
item = dataset[0]
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()
print(f'episode {episode_index}, start {from_idx}, end {to_idx}')
print(item.keys())
print(item["action"])
print(item["observation.state"])

fps =  dataset.meta.fps
print(f'fps {fps}')

frames = []
camera_key = dataset.meta.camera_keys[1]
for idx in range(from_idx, to_idx):
    original_frame = dataset[idx][camera_key]
    pil_image = to_pil(original_frame)
    frames.append(np.array(pil_image))
imageio.mimsave("franka_render_test.mp4", np.array(frames), fps=fps)
import collections
import dataclasses
import logging
import math
import pathlib
import os
import time

import cv2
import draccus
import imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_infer_perf():
    # --- Load Policy ---
    policy_path: str = "/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0828/checkpoints/050000/pretrained_model"
    policy_path = "/opt/projects/news/lerobot/outputs/train/autodl-step1/060000/pretrained_model"
    device: str = "cuda"
    seed = 8
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    policy: SmolVLA2Policy = SmolVLA2Policy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()
    
    policy.reset()

    # mock data
    observation = {
        "observation.images.image": torch.from_numpy(np.random.randint(size=(1, 3, 256, 256), low=0, high=256, dtype=np.uint8) / 255.0) .to(torch.float32).to(device),
        "observation.images.wrist_image": torch.from_numpy(np.random.randint(size=(1, 3, 256, 256), low=0, high=256, dtype=np.uint8) / 255.0).to(torch.float32).to(device),
        "observation.state": torch.from_numpy(np.array([[-0.2, 0, 1.17, 3.13, 0, 0, 0.03878, -0.03878]])).to(torch.float32).to(device),
        "task": "pick cup and place",
    }
    
    for i in range(100):
        with torch.inference_mode():
            start = time.time()
            action_tensor = policy.select_action(observation)
            end = time.time()
            print(f"Inference time: {(end-start) * 1000 :.2f}ms")
    
if __name__ == "__main__":
    test_infer_perf()
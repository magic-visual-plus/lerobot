import dataclasses
from lerobot.policies.pretrained import PreTrainedPolicy
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

from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot.policies.smolvla3.modeling_smolvla3 import SmolVLA3Policy
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy
from lerobot.policies.pretrained import PreTrainedPolicy

np.set_printoptions(precision=3, suppress=True)

NEED_FIRST_FRAME = True
LIBERO_ENV_RESOLUTION = 256


policy_version_map: dict[str, Any] = {
    "v2" : SmolVLA2Policy,
    "v3" : SmolVLA3Policy,
    "v4" : SmolVLA4Policy,
}

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    # --- Hugging Face arguments ---
    policy_path: str = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/only_bbox/pretrained_model_2w"

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_goal"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "/opt/projects/xbkaishui/lerobot/data/libero/1020/new_goal_autodl_only_bbox"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""
    # 预测的版本
    version: str = "v4"
    
def init_policy(args: Args ):
    print(f'args version {args.version}')
    policy_clazz = policy_version_map[args.version]
    policy = policy_clazz.from_pretrained(args.policy_path)
     # Handle accelerate-wrapped models by unwrapping them
    if hasattr(policy, 'module') and isinstance(policy.module, PreTrainedPolicy):
        print("got accelerate model")
        # This is likely an accelerate-wrapped model (DistributedDataParallel)
        policy: PreTrainedPolicy = policy.module
     
    print(f'n_action_steps:{policy.config.n_action_steps}')
    policy.config.n_action_steps = 8
    print(f'after reset n_action_steps:{policy.config.n_action_steps}')
    policy.to(args.device)
    policy.eval()
    return policy

@draccus.wrap()
def replay_and_eval_bbox(args: Args):
    task_id = 5
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # --- Load Policy ---
    policy: PreTrainedPolicy = init_policy(args)
    
    ipdb.set_trace()

    to_pil = transforms.ToPILImage()

    dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1019/new_goal_autodl_add_point_5w/libero_eval_20251020_154021'
    
    # dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1019/new_goal_autodl_add_point_5w_2/libero_eval_20251020_165848'

    dataset = LeRobotDataset(dataset_path)
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    total_episodes = dataset.num_episodes
    print(f'total_episodes {total_episodes}')
    
    os.makedirs(args.video_out_path, exist_ok=True)
     
    for episode_idx in range(total_episodes):
        # todo
        # episode_idx = episode_idx + 1 
        tasks = dataset.meta.episodes[episode_idx]["tasks"][0]
        
        # print(f"{dataset[episode_idx]['observation.state'].shape=}")  # (6, c)
        from_idx = dataset.episode_data_index["from"][episode_idx].item()
        to_idx = dataset.episode_data_index["to"][episode_idx].item()
        print(f"episode idx {episode_idx}, task {tasks}, from idx {from_idx} to idx {to_idx}")
        
        frames = []
        for frame_index in np.arange(from_idx, to_idx).tolist():
            frame = dataset[frame_index]
            bboxes = frame['bboxes']
            # print(f"bboxes {bboxes.shape}")
            state = frame['observation.state']
            task_description = tasks
            agentview_image_bbox = to_pil(frame['observation.images.image'])
            agentview_image_bbox = np.asarray(agentview_image_bbox).copy()
            # compose obs and send to policy
            observation = {
                "observation.images.image": frame['observation.images.image']
                .to(args.device).unsqueeze(0),
                "observation.images.wrist_image": frame['observation.images.wrist_image']
                .to(args.device).unsqueeze(0),
                "observation.state": state.to(args.device).unsqueeze(0),
                "task": task_description,
            }
            
            with torch.inference_mode():
                action_result_tensor = policy.select_action(observation, need_bbox = True)
                bbox = action_result_tensor['box']
                
                box0 = bbox[0, 0, :].cpu().numpy()
                box0 = (box0 * LIBERO_ENV_RESOLUTION).astype(int)
                box0 = box0.tolist()
            
                cv2.rectangle(
                    agentview_image_bbox,
                    (box0[0], box0[1]),
                    (box0[0]+box0[2], box0[1]+box0[3]),
                    (0, 255, 0),
                    2,
                )
                
                if 'point' in action_result_tensor:
                    points = action_result_tensor['point']
                    # first point
                    first_point: Any = points[0, 0, :].cpu().numpy()
                    first_point = (first_point * LIBERO_ENV_RESOLUTION).astype(int)
                    cv2.circle(agentview_image_bbox, (first_point[0], first_point[1]), 2, (0, 0, 255), -1)
                
                frames.append(agentview_image_bbox)
                
                cv2.imwrite("/tmp/agentview_image_bbox.png", agentview_image_bbox)
                
                ipdb.set_trace()
                
                if NEED_FIRST_FRAME:
                    print(f"early exit caused by first frame flag")
                    sys.exit(1)

        # write to video out
        video_path = (
            pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}.mp4"
        )
        fps = 20
        writer = imageio.get_writer(video_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info(f"Saved video to {video_path}") 


if __name__ == "__main__":
    replay_and_eval_bbox()
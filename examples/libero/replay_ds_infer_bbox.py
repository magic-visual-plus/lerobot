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

from torchvision import transforms

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from lerobot.policies.smolvla3.modeling_smolvla3 import SmolVLA3Policy
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy
from lerobot.policies.pretrained import PreTrainedPolicy

np.set_printoptions(precision=3, suppress=True)

policy_version_map: dict[str, Any] = {
    "v2" : SmolVLA2Policy,
    "v3" : SmolVLA3Policy,
    "v4" : SmolVLA4Policy,
}

LIBERO_ENV_RESOLUTION = 256

NEED_FIRST_EPISODE = False
DISABLE_STATE = False
DISABLE_IMAGE = False
DISABLE_WRIST_IMAGE = False
DISABLE_LANG = False

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """
    # --- Hugging Face arguments ---
    # policy_path: str = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/only_bbox/pretrained_model_2w"
    # policy_path: str = "/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1020/disable_action/pretrained_model_2w"
    
    # disable box embd
    # policy_path:str = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1022/libero_smolvla4_1022_goal_autodl_disable_bbox_emb/pretrained_model_2w'
    
    # freeze vision encoder
    # policy_path: str = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1023/libero_smolvla4_1023_goal_autodl_disable_bbox_emb_vit_encoder_action/pretrained_model_5k'
    
    policy_path: str = '/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1023/libero_smolvla4_1023_goal_autodl_disable_bbox_emb_vit_encoder_action/pretrained_model_2w'
    # adj weight
    #  policy_path:str ='/opt/projects/xbkaishui/lerobot/ckpts/smol4/goal/1022/libero_smolvla4_1022_goal_autodl_action_pretrain_bbox_aj_weight/pretrained_model_3w'

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_goal"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 10
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    # video_out_path: str = "/opt/projects/xbkaishui/lerobot/data/libero/1022/disable_bbox_emb_2w"
    video_out_path: str = "/opt/projects/xbkaishui/lerobot/data/libero/1023/disable_bbox_emb_vit_encoder_action_2w"
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
    # ipdb.set_trace()
    return policy

@draccus.wrap()
def replay_and_eval_bbox(args: Args):
    task_id = 5
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # --- Load Policy ---
    policy = init_policy(args)

    to_pil = transforms.ToPILImage()

    # dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1019/new_goal_autodl_add_point_5w/libero_eval_20251020_154021'
    dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1021/new_goal_autodl_with_bbox_action_randpos/libero_eval_20251022_002541'
    
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
        if episode_idx not in [0,1,2,3,4,5]:
            print(f"Processing episode {episode_idx}")
            continue
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
            agentview_image = frame['observation.images.image'].to(args.device).unsqueeze(0)
            wrist_image = frame['observation.images.wrist_image'].to(args.device).unsqueeze(0)
            
            if DISABLE_IMAGE:
                agentview_image = torch.zeros_like(agentview_image, device=args.device)
            if DISABLE_WRIST_IMAGE:
                wrist_image = torch.zeros_like(wrist_image, device=args.device)
            
            state = state.to(args.device).unsqueeze(0)
            if DISABLE_STATE:
                state = torch.zeros_like(state, device=args.device)
            if DISABLE_LANG:
                task_description = ""
            # TODO disable later task_description
            # task_description = "plate"
            # compose obs and send to policy
            observation = {
                "observation.images.image": agentview_image,
                "observation.images.wrist_image": wrist_image,
                "observation.state": state,
                "task": task_description,
            }
            
            # skip too many frames
            # if frame_index % 2 != 0:
                # continue
            
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
            
            # Filter out rows where the second column (index 1) has value 0
            # Assuming bboxes is a 2D array where each row is a 6D bounding box
            if len(bboxes) > 0:
                # Create a mask for rows where the second column is NOT 0
                filtered_mask = bboxes[:, 1] != 0
                filtered_bboxes = bboxes[filtered_mask]
                # Optionally, you can also show which rows were filtered out
                if filtered_bboxes.shape[0] < bboxes.shape[0]:
                    zero_rows = bboxes[bboxes[:, 1] == 0]
                    # print(f"Removed {zero_rows.shape[0]} bounding boxes with second column value = 0")
                    # print("Removed bounding boxes:")
                    print(zero_rows)
                else:
                    print("not zero box class..........")
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
        
        if NEED_FIRST_EPISODE:
            sys.exit(1)

if __name__ == "__main__":
    replay_and_eval_bbox()
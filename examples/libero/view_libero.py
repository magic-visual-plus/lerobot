"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
https://github.com/huggingface/lerobot/issues/1316
"""

import collections
import dataclasses
import logging
import math
import pathlib
import os
import time

# import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
import cv2


from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data



def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """

    # --- Hugging Face arguments ---
    # policy_path: str = "lerobot/smolvla_base"
    policy_path: str = "/opt/projects/news/lerobot/outputs/train/libero_smolvla2_0828/checkpoints/050000/pretrained_model"
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_goal"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "data/libero/test_states"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""
    
    task_index: int = 0


@draccus.wrap()
def view_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Available options are: {list(benchmark_dict.keys())}"
        )
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    task_index = args.task_index
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(min(args.num_trials_per_task, len(initial_states))),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and policy
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Add initial frame
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            # frames.append(agentview_image)
            # import ipdb; ipdb.set_trace()
            cv2.imwrite(os.path.join(args.video_out_path, f"{task_id}_{episode_idx}.png"), agentview_image)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation_log.txt"),
            logging.StreamHandler()  # Optional: keeps logging in the terminal too
        ]
    )
    view_libero()
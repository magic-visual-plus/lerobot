"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
https://github.com/huggingface/lerobot/issues/1316
"""

import collections
import dataclasses
import logging
import math
import multiprocessing

from multiprocessing.pool import Pool
import time
import math
from multiprocessing import Pool, cpu_count
import pathlib
import os
import time
import copy

import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import sys

def get_appropriate_start_method():
    """根据平台选择合适的启动方法"""
    if sys.platform == 'win32':
        return 'spawn'  # Windows 只支持 spawn
    else:
        # Unix-like 系统可以根据需要选择
        # return 'fork'    # 默认，快速但可能有线程安全问题
        # return 'spawn'   # 更安全但稍慢
        return 'forkserver'  # 折中方案，专用服务器进程管理fork
    
    

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
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 50
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "data/libero/videos"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""
    
    env_size: int = 10
    # 当前要评估的task id
    task_id: int = 0

@dataclass
class EvalResult:
    task_id: int
    metrics: Dict[str, Any]
    success: bool
    error: Optional[str] = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def init_policy(args: Args):
    policy = SmolVLA2Policy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()
    return policy

def detect_max_steps(args: Args) -> int:
    """
    Detect the maximum number of steps for each task.
    """
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        # Fallback for custom task suites
        max_steps = 520
    return max_steps
    
# @draccus.wrap()
def eval_libero(args: Args) -> EvalResult:
    # Set random seed
    set_seed(args.seed)
    # --- Load Policy ---
    policy: SmolVLA2Policy = init_policy(args)

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
    
    max_steps: int = detect_max_steps(args)
    logging.info(f"Max steps: {max_steps}")
    task_episodes, task_successes = 0, 0
    task_id = args.task_id
    # for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
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
        policy.reset()

        # Set initial states
        obs = env.set_init_state(initial_states[episode_idx])

        # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        # and we need to wait for them to fall
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        # Setup
        t = 0
        frames = []
        done = False

        # Add initial frame
        agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        # frames.append(agentview_image)
        # import ipdb; ipdb.set_trace()
        logging.info(f"Starting episode {task_episodes+1}...")
        while t < max_steps:
                try:
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.images.image": torch.from_numpy(agentview_image / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.images.wrist_image": torch.from_numpy(wrist_img / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }
                    
                    # import ipdb; ipdb.set_trace()
                    # Query model to get action
                    with torch.inference_mode():
                        start = time.time()
                        action_tensor = policy.select_action(observation)
                        end = time.time()
                        # logging.info(f"Inference time: {(end-start) * 1000 :.2f}ms")
                    action = action_tensor.cpu().numpy()[0]
                    # action[-1] = 1 - action[-1]
                    action = normalize_gripper_action(action, binarize=False)
                    action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, _, done, _ = env.step(action)
                    if done:
                        task_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

        task_episodes += 1
        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_").replace("/", "_")
        video_path = (
            pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
        )
        fps = 10
        writer = imageio.get_writer(video_path, fps=fps)

        for image in frames:
            writer.append_data(image)
        writer.close()
        logging.info(f"Saved video to {video_path}")
        # import ipdb; ipdb.set_trace()
        if task_episodes > 0:
            logging.info(f"# task {task_id} episodes completed so far: {task_episodes}")
            logging.info(f"# task {task_id} successes: {task_successes} ({task_successes / task_episodes * 100:.1f}%)")

    # Log final results for the task
    if task_episodes > 0:
        logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}")
        
    return EvalResult(task_id=task_id, metrics={'task_successes':task_successes,'task_episodes':task_episodes}, success=True)


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


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


@draccus.wrap()
def batch_eval(args: Args) -> None:
    start = time.time()
    logging.info(f"Starting batch eval, *args = {args}")
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
    logging.info(f"task suite name {args.task_suite_name} total task suite: {num_tasks_in_suite}")
    logging.info(f"total env {args.env_size}")
    
    # init process pools size is env_size 
    pool =  Pool(processes=args.env_size)
    result_futures = []
    # TODO disable later
    # num_tasks_in_suite = 1
    for i in range(num_tasks_in_suite):
        args_cp = copy.deepcopy(args)
        args_cp.task_id = i
        # eval_libero(args_cp)
        result = pool.apply_async(eval_libero, (args_cp,))
        result_futures.append(result)
        
    # get all result 
    results: list[EvalResult] = [result.get() for result in result_futures]
    # aggregate final result 
    total_sucess = 0.0
    total_eposide = 0.0
    for result in results:
        total_sucess += result.metrics["task_successes"]
        total_eposide += result.metrics["task_episodes"]
    
    logging.info(f"all task success rate: {float(total_sucess) / float(total_eposide):.2f}")
    end = time.time()
    logging.info(f"total time: {end - start:.2f}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation_batch_log.txt"),
            logging.StreamHandler()  # Optional: keeps logging in the terminal too
        ]
    )
    
    start_method = get_appropriate_start_method()
    logging.info(f"processor Using start method: {start_method}")
    multiprocessing.set_start_method(start_method)
    batch_eval()

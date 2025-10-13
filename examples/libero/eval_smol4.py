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
from sys import version
import time
from typing import Any
from loguru import logger

# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_GL"] = "osmesa"

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from lerobot.policies.smolvla3.modeling_smolvla3 import SmolVLA3Policy
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
from lerobot.policies.smolvla4.modeling_smolvla4 import SmolVLA4Policy
from lerobot.policies.pretrained import PreTrainedPolicy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# NEED_RAND_POS = True
NEED_RAND_POS = False
NEED_BBOX = True
NEED_RAND_CAM = False

DISABLE_STATE = True
DISABLE_IMAGE = False
DISABLE_WRIST_IMAGE = False
DISABLE_PROMPT = False

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
    # 预测的版本
    version: str = "v4"


policy_version_map: dict[str, Any] = {
    "v2" : SmolVLA2Policy,
    "v3" : SmolVLA3Policy,
    "v4" : SmolVLA4Policy,
}

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
    # policy.config.n_action_steps = 8
    print(f'after reset n_action_steps:{policy.config.n_action_steps}')
    policy.to(args.device)
    policy.eval()
    return policy


@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Policy ---
    policy = init_policy(args)

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

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    # TODO disable later
    # num_tasks_in_suite = 1
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        if task_id != 5:
            continue
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
            desc=f"Task {task_id}: {task.language}, {task_successes}/{task_episodes}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and policy
            env.reset()
            policy.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])
            
            if NEED_RAND_CAM:
                # todo 修改相机位置
                # print(f"original cam pos: {env.env.sim.model.cam_pos}")
                cam_pos = env.env.sim.model.cam_pos[2].copy()
                # 在每个方向上随机偏移（单位：米）
                x_offset = np.random.uniform(-0.05, 0.05)
                y_offset = np.random.uniform(-0.05, 0.05)
                z_offset = np.random.uniform(-0.03, 0.03)
                # 应用偏移
                cam_pos += np.array([x_offset, y_offset, z_offset])
                # 写回模型
                env.env.sim.model.cam_pos[2] = cam_pos

                # 当前相机的四元数
                quat = env.env.sim.model.cam_quat[2]
                quat_scipy = np.roll(quat, -1)  # 转为 [x, y, z, w]
                # 随机欧拉角扰动（单位：度）
                pitch_offset = np.random.uniform(-3, 3)  # 上下抬头
                yaw_offset = np.random.uniform(-3, 3)    # 左右旋转
                roll_offset = np.random.uniform(-3, 3)   # 轻微倾斜
                rot = R.from_quat(quat_scipy)
                rot_offset = R.from_euler('xyz', [pitch_offset, yaw_offset, roll_offset], degrees=True)
                new_rot = rot * rot_offset
                # 写回到 MuJoCo（注意格式为 [w, x, y, z]）
                env.env.sim.model.cam_quat[2] = np.roll(new_rot.as_quat(), 1)
            
            if NEED_RAND_POS:
                # set object name 
                object_name = "plate_1"
                plate_object = env.env.get_object(object_name)
                logging.info(f'plage object {plate_object}')
                pos = env.env.get_object_position(object_name)
                logging.info(f"object_name {object_name}, 位置: {pos}")  # [x, y,
                # 2. 获取对象姿态（位置+旋转）
                pose = env.env.get_object_pose(object_name) 
                logging.info(f"object_name {object_name}, 四元数: {pose['quat']}")
                workspace_offset = env.env.get_workspace_offset()
                logging.info(f'workspace offset {workspace_offset}')
                # random x offset 
                x_offset = np.random.uniform(0.03, 0.1)
                y_offset = np.random.uniform(0.03, 0.1)
                new_pos = pos + [x_offset, y_offset, -0.9]
                logging.info(f'set object_name {object_name}, old pos {pos}, new pos {new_pos}')
                # TODO enable later
                env.env.set_object_position(object_name, new_pos.tolist(), quaternion=pose['quat'])

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
                    agentview_image_bbox = agentview_image.copy()
                    frames.append(agentview_image)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    
                    if DISABLE_STATE:
                        # disable state to zeros
                        # state = np.zeros_like(state)
                        # fix state most right 
                        state = np.array([-0.06813535, -0.19478797,  1.16284806,  3.099479,   -0.01742949, -0.80769123, 0.01611037, -0.03959287])
                        # fix state most left
                        state = np.array([-1.68719120e-03,  2.43636704e-01,  1.02261244e+00, 3.099479,   -0.01742949, -0.80769123, 0.01611037, -0.03959287])

                    if DISABLE_IMAGE:
                        agentview_image = np.zeros_like(agentview_image)
                        
                    if DISABLE_WRIST_IMAGE:
                        wrist_img = np.zeros_like(wrist_img)
                        
                    if DISABLE_PROMPT:
                        task_description = "please description the image and do something"
                    
                    # print(f"robot state {state}")
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
                        action_tensor = policy.select_action(observation, need_bbox=NEED_BBOX)
                        if NEED_BBOX:
                            bbox = action_tensor['box']
                            action_tensor = action_tensor['action']
                            # logging.info(f"action_tensor {action_tensor}")
                            # draw box
                            box0 = bbox[0, 0, :].cpu().numpy()
                            box0 = (box0 * LIBERO_ENV_RESOLUTION).astype(int)
                            cv2.rectangle(
                                agentview_image_bbox,
                                (box0[0], box0[1]),
                                (box0[0]+box0[2], box0[1]+box0[3]),
                                (0, 255, 0),
                                2,
                            )
                            frames.append(agentview_image_bbox)
                        else:
                            ...
                            # logging.info(f"action_tensor {action_tensor}")
                        end = time.time()
                        # logging.info(f"Inference time: {(end-start) * 1000 :.2f}ms")
                    action = action_tensor.cpu().numpy()[0]
                    # print(f"Step {t}: action = {action}")
                    # action[-1] = 1 - action[-1]
                    action = normalize_gripper_action(action, binarize=False)
                    action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, _, done, _ = env.step(action)
                    print(f"step {t}, state {state} . done status {done}")
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logger.exception(e)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = (
                pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{task_segment}_{suffix}.mp4"
            )
            fps = 20
            writer = imageio.get_writer(video_path, fps=fps)

            for image in frames:
                writer.append_data(image)
            writer.close()
            logging.info(f"Saved video to {video_path}")
            # import ipdb; ipdb.set_trace()

            # Log current results
            logging.info(f"Success: {done}")
            if total_episodes > 0:
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results for the task
        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.2f}")
        if total_episodes > 0:
            logging.info(f"Cumulative success rate: {float(total_successes) / float(total_episodes):.2f}")
        # close env
        env.close()
        del env

    logging.info("--- Evaluation finished ---")
    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.2f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    # cv2.destroyAllWindows()


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation_log.txt"),
            logging.StreamHandler()  # Optional: keeps logging in the terminal too
        ]
    )
    eval_libero()
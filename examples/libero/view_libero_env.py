from numpy._typing._array_like import NDArray


from numpy._typing._array_like import NDArray


import numpy as np
from loguru import logger
import os 

import collections
import dataclasses
import logging
import math
import pathlib
import time
from typing import Any
import cv2

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.datasets.rotation_transformer import RotationTransformer
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from scipy.spatial.transform import Rotation as R


LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
# LIBERO_DUMMY_ACTION = [0.5, 0.5, -0.46, 0.0, 0.0, 0.0, -1.0]
LIBERO_DUMMY_ACTION = [1, 1.5, -0.46, 0.0, 0.0, 0.0, -1.0]



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


def save_one_image(obs, image_name):
    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    cv2.imwrite(image_name, agentview_image)

def view_libero_env() -> None:
    task_suite_name = "libero_goal"
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[task_suite_name]()
    except KeyError:
        raise ValueError(
        f"Unknown task suite: {task_suite_name}. "
        f"Available options are: {list(benchmark_dict.keys())}"
    )
    num_tasks_in_suite = task_suite.n_tasks
    task_idx_dict = {task.name: i for i, task in enumerate(task_suite.tasks)}
    print(f'Task suite: {task_suite_name}, tasks {task_idx_dict}')
    task_id = 5
    episode_idx = 0 
    
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, 0)
    robot_rand_state: NDArray[Any] = np.array([0.05386879, -0.02057877,  0.92886889,  3.13609323,  0.02380274, -0.20587286, 0.03983286, -0.03983748])
    env.reset()
    # Set initial states
    obs = env.set_init_state(initial_states[episode_idx])
    # and we need to wait for them to fall
    for _ in range(20):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    
    time.sleep(1)
    # env.set_state()
    save_one_image(obs, f"test_{task_id}_{episode_idx}_init.png")
    print(f'int obs {obs["robot0_eef_pos"]}')
    
    state = np.concatenate(
        (
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
        )
    )
    
    print(f'state {state}')

    # env step
    # obs, _, _, _ = env.step(np.array([0.2, 0.3, -0.26, 0.0, 0.0, 0.0, -1.0]))
    # time.sleep(1)
    # print(f'after step obs {obs["robot0_eef_pos"]}')
    # save_one_image(obs, f"test_{task_id}_{episode_idx}_move_left.png")
    env.close()
    del env

if __name__ == "__main__":
    view_libero_env()
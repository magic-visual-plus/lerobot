import numpy as np
from reply_train_config import TASK_CONFIGS
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla2.modeling_smolvla2 import SmolVLA2Policy
import datetime
import os
import torch
import cv2
import math
from lerobot.policies.pretrained import PreTrainedPolicy
import dataclasses

LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
        "fps": 20,
        "info": {
            "video.height": 256,
            "video.width": 256,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 20,
            "video.channels": 3,
            "has_audio": False
        }
    },
    "observation.images.depth": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
    },
    "observation.images.segmentation": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_depth": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_segmentation": {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]
        },
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    # "observation.states.joint_state": {
    #     "dtype": "float32",
    #     "shape": (7,),
    #     "names": {
    #         "motors": [
    #             "joint_0",
    #             "joint_1",
    #             "joint_2",
    #             "joint_3",
    #             "joint_4",
    #             "joint_5",
    #             "joint_6",
    #         ]
    #     },
    # },
    "observation.mj_state": {
        "dtype": "float32",
        "shape": (200,),
        "names": None
    },
    # "observation.states.gripper_state": {
    #     "dtype": "float32",
    #     "shape": (2,),
    #     "names": {"motors": ["gripper", "gripper"]},
    # },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
    "bboxes": {
        "dtype": "float32",
        "shape": (10, 6),
        "names": {"motors": ["class", "x", "y", "w", "h"]},
    },
    "wrist_bboxes": {
        "dtype": "float32",
        "shape": (10, 6),
        "names": {"motors": ["class", "x", "y", "w", "h"]},
    },
    "timestamp": {
        "dtype": "float32",
        "shape": [
            1
        ],
        "names": None
    },
}

remove_feature = [
    "index",
    "task",
    "episode_index",
    "task_index",
    "timestamp",
    "frame_index",
]


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

def depth_to_uint8(depth, depth_min, depth_max):
    """
    将深度数组归一化到 [0,1]，然后乘以 255 转为 uint8
    
    Args:
        depth (np.ndarray): 深度数组
        depth_min (float): 深度最小值
        depth_max (float): 深度最大值
    
    Returns:
        np.ndarray: uint8 深度图
    """
    depth_norm = (depth.astype(np.float32) - depth_min) / (depth_max - depth_min)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    return (depth_norm * 255).astype(np.uint8)

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
    video_out_path: str = "data/libero/videos"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 7
    """Random Seed (for reproducibility)"""

def init_policy(args: Args):
    policy = SmolVLA2Policy.from_pretrained(args.policy_path)
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

def bboxes_to_matrix(
    bboxes,
    id_map=None,
    img_width=255,
    img_height=255,
    max_boxes=10
):
    """
    bboxes: [{'id': int, 'name': str, 'bbox': [x, y, w, h]}, ...]
    id_map: {id: extra_value, ...}  # 通过id映射到一个额外的值
    返回 shape=(max_boxes, 6) 的矩阵 [id, mapped_value, x, y, w, h]
    """
    if id_map is None:
        id_map = {}

    # 6列：id、mapped_value、x、y、w、h
    mat = np.zeros((max_boxes, 6), dtype=np.float32)
    mat[:, 0] = -1.0      # id 默认 -1
    mat[:, 1] = -1.0      # mapped_value 默认 -1

    for i, box in enumerate(bboxes[:max_boxes]):
        x, y, w, h = box["bbox"]
        bid = box["id"]
        mat[i, 0] = bid
        # 如果id存在映射表中则取值，否则-1
        mat[i, 1] = id_map.get(bid, -1.0)
        mat[i, 2] = x / img_width
        mat[i, 3] = y / img_height
        mat[i, 4] = w / img_width
        mat[i, 5] = h / img_height

    return mat

def draw_segmentation_bboxes(seg_image, gray_to_label, label_names, tol=5, min_area=10):
    """
    在 segmentation 图上生成 COCO 风格的 bbox，
    允许像素值在目标值±tol范围内匹配，只保留每个类别中面积最大的区域。
    """
    import cv2
    import numpy as np

    bboxes = []

    for gray_val, label_id in gray_to_label.items():
        # 容差范围内都算
        mask = ((seg_image >= gray_val - tol) & (seg_image <= gray_val + tol)).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 去掉背景（第0个）
        if num_labels <= 1:
            continue

        # 找出面积最大的连通区域（忽略太小的）
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas)
        area = areas[max_idx]
        if area < min_area:
            continue

        # 取该区域的 bbox
        i = max_idx + 1  # 因为 stats[0] 是背景
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        bboxes.append({
            "id": label_id,
            "name": label_names[label_id],
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    return bboxes

class LeRobotEvalSave:
    def __init__(self, task_suite_name, base_path="/opt/projects/news/lerobot/data/eval", repo_id="new_goal_bbox",  fps=20):
        self.base_path = base_path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_ds_path = os.path.join(self.base_path, f"libero_eval_{timestamp}")
        print(f'new ds path {new_ds_path}')
        self.new_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=new_ds_path,
            fps=fps,
            robot_type="franka",
            features=LIBERO_FEATURES,
        )
        self.config = TASK_CONFIGS[task_suite_name]

        self.label_names = self.config['label_names']
        self.tasks2labels = self.config['tasks2labels']
        self.object_class = self.config['object_class']
        self.object2class = self.config['object2class']
    
        
    def init_episodes(self, task_name):
        self.gray_to_label = self.tasks2labels[task_name]
        self.object_to_class = self.object2class[task_name]
        self.frames_temp = []

    def process_obs(self, obs, action):
        # print(f'add one frame {action}')
        frame = {}
        agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
        agentview_depth = depth_to_uint8(agentview_depth, 0.98488456, 0.9965628)
        agentview_segmentation = np.ascontiguousarray(obs["agentview_segmentation_instance"][::-1, ::-1])
        frame["observation.images.image"] = agentview_image
        frame["observation.images.depth"] = cv2.cvtColor(agentview_depth, cv2.COLOR_GRAY2BGR)
        seg_uint8 = agentview_segmentation.astype(np.uint8)
        frame["observation.images.segmentation"] = cv2.cvtColor(seg_uint8, cv2.COLOR_GRAY2BGR)
        # 按时间戳保存agentview_segmentation
        # cv2.imwrite(f"agentview_segmentation_{frame_index:06d}.png", agentview_segmentation)
        seg_for_bbox = seg_uint8
        bboxes = draw_segmentation_bboxes(seg_for_bbox, self.gray_to_label, self.label_names, tol=0)
        agentview_bbox = bboxes_to_matrix(bboxes, id_map=self.object_to_class)
        # print("ddddd", agentview_bbox)
        frame["bboxes"] = agentview_bbox

        robot0_eye_in_hand_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        robot0_eye_in_hand_depth = np.ascontiguousarray(obs["robot0_eye_in_hand_depth"][::-1, ::-1])
        robot0_eye_in_hand_depth = depth_to_uint8(robot0_eye_in_hand_depth, 0.70488456, 0.9965628)
        robot0_eye_in_hand_segmentation = np.ascontiguousarray(obs["robot0_eye_in_hand_segmentation_instance"][::-1, ::-1])
        frame["observation.images.wrist_image"] = robot0_eye_in_hand_image
        frame["observation.images.wrist_depth"] = cv2.cvtColor(robot0_eye_in_hand_depth, cv2.COLOR_GRAY2BGR)
        seg_uint8 = robot0_eye_in_hand_segmentation.astype(np.uint8)
        frame["observation.images.wrist_segmentation"] = cv2.cvtColor(seg_uint8, cv2.COLOR_GRAY2BGR)  
        seg_for_bbox = seg_uint8
        bboxes = draw_segmentation_bboxes(seg_for_bbox, self.gray_to_label, self.label_names, tol=0)
        robot0_eye_in_hand_bbox = bboxes_to_matrix(bboxes, id_map=self.object_to_class)
        frame["wrist_bboxes"] = robot0_eye_in_hand_bbox

        obs_state_rpy = _quat2axisangle(obs["robot0_eef_quat"])
        obs_state_pos = obs['robot0_eef_pos']
        gripper_state = obs["robot0_gripper_qpos"]
        obs_state = np.concatenate((obs_state_pos, obs_state_rpy, gripper_state))
        obs_state_np = obs_state                        # numpy.ndarray

        # 转为 torch，匹配设备和类型
        obs_state_torch = torch.tensor(obs_state_np, dtype=torch.float32)

        # 更新回去（可选，取决于frame是否mutable）mock value
        frame['observation.state'] = obs_state_torch
        frame['observation.mj_state'] = np.ones(200, dtype=np.float32) * -999 # 占位符

        frame['action'] = torch.tensor(action, dtype=torch.float32)
        frame['observation.states.ee_state'] = torch.tensor(obs_state[:6], dtype=torch.float32)

        self.frames_temp.append(frame)
    
    def clear(self):    
        """
        clear all temp frames
        """
        print(f'clear frames')
        self.frames_temp = []
        
    def save_episode(self, task_description):
        print(f'len frames {len(self.frames_temp)}')
        for frame in self.frames_temp:
            self.new_dataset.add_frame(frame, task_description)
        self.new_dataset.save_episode()
        self.frames_temp = []
        
    def get_episode_size(self) -> int:
        if self.new_dataset.episodes is None:
            return 0
        return len(self.new_dataset.episodes)
    

if __name__ == "__main__":
    task_suite_name = 'libero_goal'
    eval_save = LeRobotEvalSave(task_suite_name)
    
    task_name = "put_the_wine_bottle_on_top_of_the_cabinet"
    eval_save.init_episodes(task_name)
    
    eposide_len = eval_save.get_episode_size()
    print(f'eposide_len: {eposide_len}')
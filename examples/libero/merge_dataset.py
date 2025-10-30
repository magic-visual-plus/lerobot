import os
import shutil
from tqdm import tqdm
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# ======================= 配置 =======================
dataset_orginal = '/opt/projects/news/lerobot/data/10.13/libero_goal_no_lerobot_0'

dataset_orginal = "/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds_origin"


dataset_to_be_added = r'/opt/projects/xbkaishui/lerobot/data/libero/1019/new_goal_autodl_add_point_5w_3/libero_eval_20251025_234929'

new_ds_path = r'/opt/projects/news/lerobot/data/10.13/goal_single_task_5_ds_mix_action'

# ======================= 准备新目录 =======================
if os.path.exists(new_ds_path):
    print(f"⚠️ 目标目录已存在，正在删除：{new_ds_path}")
    shutil.rmtree(new_ds_path)

# 直接复制 dataset1 作为基础
print(f"📦 正在复制 {dataset_orginal} 到新目录 {new_ds_path} ...")
shutil.copytree(dataset_orginal, new_ds_path)
print("✅ dataset1 已复制到新目录！")

# ======================= 打开新数据集并追加 dataset2 =======================
dataset2 = LeRobotDataset(dataset_to_be_added)
ds_meta2 = LeRobotDatasetMetadata(dataset_to_be_added)

# 注意：这里用 new_ds_path 打开刚复制好的数据集
new_dataset = LeRobotDataset(new_ds_path)

IMAGE_KEYS = [
    "observation.images.image",
    "observation.images.depth",
    "observation.images.segmentation",
    "observation.images.wrist_image",
    "observation.images.wrist_depth",
    "observation.images.wrist_segmentation",
]

remove_feature = [
    "index",
    "task",
    "episode_index",
    "task_index",
    "timestamp",
    "frame_index",
    # "observation.states.gripper_state",
    # "observation.states.joint_state"
]

def convert_images_to_hwc(frame):
    """将CHW格式图像转换为HWC"""
    new_frame = {}
    for k, v in frame.items():
        if k in IMAGE_KEYS and v.shape == (3, 256, 256):
            v = np.transpose(v, (1, 2, 0))
        new_frame[k] = v
    return new_frame

print("🔹 追加 dataset2 ...")
frame_index = 0
for episode_index, episode in enumerate(tqdm(ds_meta2.episodes, desc="Appending dataset2")):
    length = ds_meta2.episodes[episode_index]['length']
    task = ds_meta2.episodes[episode_index]['tasks'][0]
    for t in range(length):
        frame = dataset2[frame_index]
        for feat in remove_feature:
            if feat in frame:
                del frame[feat]
        frame = convert_images_to_hwc(frame)
        # print(frame['observation.images.depth'].shape)
        frame['observation.states.gripper_state'] = frame['observation.state'][-2:]
        frame['observation.states.joint_state'] = frame['observation.state'][:7]
        new_dataset.add_frame(frame, task)
        frame_index += 1
    new_dataset.save_episode()

print(f"✅ 合并完成！新的数据集已保存至：{new_ds_path}")

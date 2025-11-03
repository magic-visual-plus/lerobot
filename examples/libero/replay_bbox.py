import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# === 配置路径 ===
# dataset_path = "/opt/projects/news/lerobot/data/10.13/libero_spatial_no_lerobot_0"
dataset_path = '/opt/projects/xbkaishui/lerobot/data/libero/1019/new_goal_autodl_add_point_5w/libero_eval_20251020_154021'

save_dir = "/opt/projects/xbkaishui/lerobot/data/libero/1019/draw_bboxes_video_goal"
os.makedirs(save_dir, exist_ok=True)

# === 颜色映射（BGR）===
CLASS_COLOR = {
    0: (0, 0, 255),    # 红
    1: (0, 255, 0),    # 绿
    2: (255, 0, 0),    # 蓝
    3: (255, 255, 255) # 白
}
DEFAULT_COLOR = (0, 255, 255)  # 黄色

# === 打开数据集 ===
dataset = LeRobotDataset(dataset_path)
ds_meta = LeRobotDatasetMetadata(dataset_path)

fps = 30
frame_index = 0  # 全局帧索引（以数据集索引为准）

def tensor_to_bgr_image(t):
    """
    把 [C,H,W] 或 [H,W,C] 的张量/ndarray 转成 OpenCV BGR uint8 图像（0-255）。
    假设输入像素范围在 [0,1] 或 [0,255]。
    """
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    t = np.asarray(t)
    # 如果是 [C,H,W] -> 转为 [H,W,C]
    if t.ndim == 3 and t.shape[0] <= 4 and t.shape[0] != t.shape[2]:
        t = np.transpose(t, (1, 2, 0))
    # 若单通道，转成3通道
    if t.ndim == 2:
        t = np.stack([t] * 3, axis=-1)
    # 归一化到 0-255
    if t.max() <= 1.0:
        t = (t * 255.0).clip(0, 255).astype(np.uint8)
    else:
        t = t.clip(0, 255).astype(np.uint8)
    # 假设输入为 RGB，转换成 BGR
    if t.shape[2] == 3:
        t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    return t

def build_mask_from_bbox(bbox, H, W):
    """
    bbox 格式假设为 [obj_id, cls_id, x, y, bw, bh]
    其中 x,y 表示左上角归一化坐标（0-1），bw,bh 为宽高（归一化）。
    返回二值 mask (H, W) -- 1 inside bbox else 0
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    _, _, x, y, bw, bh = bbox
    x1 = int(x * W)
    y1 = int(y * H)
    x2 = int((x + bw) * W)
    y2 = int((y + bh) * H)
    # clamp
    x1 = max(0, min(W-1, x1))
    x2 = max(0, min(W-1, x2))
    y1 = max(0, min(H-1, y1))
    y2 = max(0, min(H-1, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask

# === 遍历 episode 并生成视频 ===
for episode_index in tqdm(range(len(ds_meta.episodes)), desc="Episodes"):
    ep = ds_meta.episodes[episode_index]
    length = int(ep["length"])

    # 读取第一帧以确定主视频大小
    item0 = dataset[frame_index]
    img0 = item0["observation.images.image"]
    img0 = tensor_to_bgr_image(img0)
    H, W = img0.shape[:2]

    # wrist 图像大小（可能与主图不同），取第一帧 wrist_image 判断
    wrist_img0 = item0.get("observation.images.wrist_image", None)
    if wrist_img0 is not None:
        wrist_img0 = tensor_to_bgr_image(wrist_img0)
        Hw, Ww = wrist_img0.shape[:2]
    else:
        Hw, Ww = None, None

    # 创建两个视频文件：主图 + wrist 图（如果存在 wrist 图）
    out_path_main = os.path.join(save_dir, f"episode_{episode_index:03d}_main.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_main = cv2.VideoWriter(out_path_main, fourcc, fps, (W, H))

    writer_wrist = None
    if Hw is not None:
        out_path_wrist = os.path.join(save_dir, f"episode_{episode_index:03d}_wrist.mp4")
        writer_wrist = cv2.VideoWriter(out_path_wrist, fourcc, fps, (Ww, Hw))

    for _ in range(length):
        item = dataset[frame_index]

        # ===== 主图像（在其上绘制 bboxes） =====
        # img = (tensor_to_bgr_image(item["observation.images.segmentation"]).copy() * 255 - 245) * 10
        img = tensor_to_bgr_image(item["observation.images.image"]).copy()

        # 读取 bboxes
        bboxes = item.get("bboxes", None)
        if bboxes is not None:
            if torch.is_tensor(bboxes):
                bboxes = bboxes.detach().cpu().numpy()
            # 支持空或形状不对的情况
            if bboxes.ndim == 2 and bboxes.shape[0] > 0:
                for box in bboxes:
                    # box: [obj_id, cls_id, x, y, bw, bh]
                    if len(box) == 5:
                        # obj_id = 0
                        cls_id = 0
                        obj_id = int(box[0])
                        x = float(box[1])
                        y = float(box[2])
                        bw = float(box[3])
                        bh = float(box[4])
                    else:
                        obj_id = int(box[0])
                        cls_id = int(box[1])
                        x = float(box[2])
                        y = float(box[3])
                        bw = float(box[4])
                        bh = float(box[5])

                    x1 = int(x * W)
                    y1 = int(y * H)
                    x2 = int((x + bw) * W)
                    y2 = int((y + bh) * H)

                    # clamp
                    x1 = max(0, min(W-1, x1))
                    x2 = max(0, min(W-1, x2))
                    y1 = max(0, min(H-1, y1))
                    y2 = max(0, min(H-1, y2))

                    color = CLASS_COLOR.get(cls_id, DEFAULT_COLOR)
                    if cls_id != 0:
                        continue  # Skip drawing for cls_id != 0
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        f"id:{obj_id} cls:{cls_id}",
                        (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        writer_main.write(img)

        # ===== wrist 图像（在其上绘制 wrist_bboxes） =====
        if writer_wrist is not None:
            wrist_img = tensor_to_bgr_image(item["observation.images.wrist_image"]).copy()

            wrist_bboxes = item.get("wrist_bboxes", None)
            if wrist_bboxes is not None:
                if torch.is_tensor(wrist_bboxes):
                    wrist_bboxes = wrist_bboxes.detach().cpu().numpy()
                if wrist_bboxes.ndim == 2 and wrist_bboxes.shape[0] > 0:
                    for box in wrist_bboxes:
                        # 格式假设同主 bbox: [obj_id, cls_id, x, y, bw, bh]
                        
                        if len(box) == 5:
                            # obj_id = 0
                            cls_id = 0
                            obj_id = int(box[0])
                            x = float(box[1])
                            y = float(box[2])
                            bw = float(box[3])
                            bh = float(box[4])
                        else:
                            obj_id = int(box[0])
                            cls_id = int(box[1])
                            x = float(box[2])
                            y = float(box[3])
                            bw = float(box[4])
                            bh = float(box[5])
                        

                        x1 = int(x * Ww)
                        y1 = int(y * Hw)
                        x2 = int((x + bw) * Ww)
                        y2 = int((y + bh) * Hw)

                        x1 = max(0, min(Ww-1, x1))
                        x2 = max(0, min(Ww-1, x2))
                        y1 = max(0, min(Hw-1, y1))
                        y2 = max(0, min(Hw-1, y2))

                        color = CLASS_COLOR.get(cls_id, DEFAULT_COLOR)
                        cv2.rectangle(wrist_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            wrist_img,
                            f"id:{obj_id} cls:{cls_id}",
                            (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            writer_wrist.write(wrist_img)

        frame_index += 1

    writer_main.release()
    if writer_wrist is not None:
        writer_wrist.release()

print("全部 episode 视频已保存到:", save_dir)

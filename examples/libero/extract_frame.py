import cv2

# 1. 打开视频文件
video_path = r"E:\download\draw_bboxes_video_goal\draw_bboxes_video_goal\episode_001_main.mp4"  # 这里换成你的视频路径
cap = cv2.VideoCapture(video_path)

# 2. 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件！")
    exit()

# 3. 读取第一帧
ret, frame = cap.read()
if not ret:
    print("无法读取第一帧！")
    cap.release()
    exit()

# 4. 保存为 PNG 文件
output_path = "bbox_first_frame.png"
cv2.imwrite(output_path, frame)

print(f"第一帧已保存为 {output_path}")

# 5. 释放资源
cap.release()

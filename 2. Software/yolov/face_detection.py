import cv2
import torch
import numpy as np
import os
import sys
from ultralytics import YOLO

# 检查CUDA是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 加载YOLOv8n模型
model_path = r'C:\Users\XG\Desktop\local_work\zero-robotic-arm\2. Software\yolov\runs\detect\train\weights\best.pt'
model = YOLO(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率为640*640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("按 'q' 键退出程序")

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法读取摄像头帧")
        break
    
    # 使用YOLOv8模型进行推理
    results = model(frame)
    
    # 在图像上绘制检测框
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # 只处理'niu_ma'类别（需要根据训练时的类别ID确定）
                if int(cls) == 0 and conf > 0.7:  # 置信度阈值设为0.7
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # 添加标签和置信度
                    label = f'niu_ma {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 显示结果帧
    cv2.imshow('YOLOv8 niu_ma Detection', frame)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
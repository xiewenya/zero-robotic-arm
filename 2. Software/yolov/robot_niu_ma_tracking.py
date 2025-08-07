import cv2
import torch
import numpy as np
import os
import sys
import time
import serial
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
import argparse  # 添加argparse库用于解析命令行参数

# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def update(self, error, dt):
        dt = min(dt, 0.1)  # 限制dt最大值，防止积分过大
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

    def reset(self):
        """重置PID控制器状态"""
        self.previous_error = 0
        self.integral = 0

def draw_detection_info(frame, x1, y1, x2, y2, conf, target_x, target_y, target_size_ratio):
    """绘制检测框和相关信息"""
    # 绘制边界框
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    # 添加标签和置信度
    label = f'niu_ma {conf:.2f}'
    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 绘制中心点
    cv2.circle(frame, (int(target_x), int(target_y)), 5, (0, 0, 255), -1)
    # 显示偏差信息
    cv2.putText(frame, f'X: {target_x:.1f}, Y: {target_y:.1f}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Size ratio: {target_size_ratio:.3f}', 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_control_info(frame, control_x, control_y, control_z):
    """绘制控制信息"""
    cv2.putText(frame, f'Control X: {control_x:.2f}', 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Control Y: {control_y:.2f}', 
               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f'Control Z: {control_z:.2f}', 
               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def send_control_command(ser, control_x, control_y, control_z):
    """发送控制指令到机械臂"""
    try:
        control_z_up = max(0.0, control_z)
        control_z_down = max(0.0, -control_z)
        # 构造控制指令
        command = f"remote_event {control_x:.3f} {control_y:.3f} {control_z_up:.3f} {control_z_down:.3f} 0.000 0.000\n"
        ser.write(command.encode())
        print(f"发送控制指令: {command.strip()}")
        return True
    except Exception as e:
        print(f"发送指令失败: {e}")
        return False

def update_plot(time_values, pid_x_values, pid_y_values, pid_z_values, 
                line_x, line_y, line_z, ax, current_time, start_time, 
                control_x, control_y, control_z):
    """更新PID输出图表"""
    # 更新PID输出值用于绘图
    pid_x_values.append(control_x)
    pid_y_values.append(control_y)
    pid_z_values.append(control_z)
    time_values.append(current_time - start_time)
    
    # 更新图表数据
    line_x.set_data(time_values, pid_x_values)
    line_y.set_data(time_values, pid_y_values)
    line_z.set_data(time_values, pid_z_values)
    
    # 自动调整x轴范围
    if len(time_values) > 1:
        ax.set_xlim(max(0, time_values[-1] - 10), time_values[-1] + 1)
    
    # 重绘图表
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def initialize_plot():
    """初始化PID输出图表"""
    # 设置实时绘图
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()
    ax.set_title('PID Outputs')
    ax.set_xlabel('Time')
    ax.set_ylabel('Output')
    ax.set_ylim(-1.5, 1.5)

    # 使用deque存储最近的PID输出值，限制长度为100
    pid_x_values = deque(maxlen=100)
    pid_y_values = deque(maxlen=100)
    pid_z_values = deque(maxlen=100)
    time_values = deque(maxlen=100)

    # 绘制三条线：X(红色)、Y(绿色)、Z(蓝色)
    line_x, = ax.plot([], [], 'r-', label='X PID Output')
    line_y, = ax.plot([], [], 'g-', label='Y PID Output')
    line_z, = ax.plot([], [], 'b-', label='Z PID Output')
    ax.legend()
    
    return fig, ax, line_x, line_y, line_z, pid_x_values, pid_y_values, pid_z_values, time_values

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Robot Niu Ma Tracking')
    parser.add_argument('--port', type=str, default='COM7', help='Serial port for robot communication (default: COM7)')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of PID outputs')
    args = parser.parse_args()
    
    uart_port = args.port  # 根据程序运行入参修改，默认为COM7
    enable_plot = args.plot  # 是否画图根据入参决定，默认不画图
    
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    print(f"串口端口: {uart_port}")
    print(f"绘图功能: {'启用' if enable_plot else '禁用'}")

    # 加载YOLOv8n模型
    model_path = r'.\runs\detect\train\weights\best.pt'
    model = YOLO(model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 摄像头分辨率为4:3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 图像中心点
    center_x = 320
    center_y = 240

    uart_port = 'COM7' # 根据实际串口进行修改

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("按 'q' 键退出程序")
    print("摄像头分辨率已设置为640*480")

    # PID控制器参数
    pid_x = PIDController(kp=0.001, ki=0.0, kd=0)
    pid_y = PIDController(kp=0.001, ki=0.0, kd=0)
    pid_z = PIDController(kp=0.001, ki=0.0, kd=0)

    # 目标框占图像的比例
    target_ratio = 1.0 / 10

    # 上次更新时间
    last_time = time.time()

    # 尝试连接到机械臂串口 (根据实际串口进行修改)
    try:
        ser = serial.Serial(uart_port, 115200, timeout=1)  # 根据实际串口修改
        print("成功连接到机械臂串口")
        ser.write(b"remote_enable\n") # 发送远程使能命令
    except Exception as e:
        print(f"无法连接到机械臂串口: {e}")
        ser = None

    # 初始化图表
    if enable_plot:
        fig, ax, line_x, line_y, line_z, pid_x_values, pid_y_values, pid_z_values, time_values = initialize_plot()
        # 记录开始时间用于绘图
        start_time = time.time()
    else:
        fig, ax, line_x, line_y, line_z, pid_x_values, pid_y_values, pid_z_values, time_values = None, None, None, None, None, None, None, None
        start_time = None

    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取摄像头帧")
            break
        
        # 记录时间用于PID计算
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # 使用YOLOv8模型进行推理
        results = model(frame)
        
        # 是否检测到niu_ma
        detected = False
        target_x = center_x
        target_y = center_y
        target_size_ratio = target_ratio
        
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
                        detected = True
                        
                        # 计算边界框中心点
                        target_x = (x1 + x2) / 2
                        target_y = (y1 + y2) / 2
                        
                        # 计算边界框面积占整个图像的比例
                        box_area = (x2 - x1) * (y2 - y1)
                        image_area = 640 * 640
                        target_size_ratio = box_area / image_area
                        
                        # 绘制检测信息
                        draw_detection_info(frame, x1, y1, x2, y2, conf, target_x, target_y, target_size_ratio)
        
        # 初始化控制输出为0
        control_x = 0.0
        control_y = 0.0
        control_z = 0.0
        
        # 如果检测到niu_ma，则进行追踪控制
        if detected and dt > 0:
            # 计算偏差
            error_x = target_x - center_x  # X轴偏差
            error_y = target_y - center_y  # Y轴偏差
            error_z = target_size_ratio - target_ratio  # 大小偏差
            
            # 使用PID控制器计算控制输出
            control_x = pid_x.update(error_x, dt)
            control_y = pid_y.update(error_y, dt)
            control_z = pid_z.update(error_z, dt)
                    
            # 限制控制输出范围
            control_x = np.clip(control_x, -1.0, 1.0)
            control_y = np.clip(control_y, -1.0, 1.0)
            control_z = np.clip(control_z, -1.0, 1.0)
            
            # 发送控制指令到机械臂
            if ser:
                send_control_command(ser, control_x, control_y, control_z)
        else:
            # 重置PID控制器的积分和上一次误差，防止I/D输出波动较大
            # pid_x.reset()
            # pid_y.reset()
            # pid_z.reset()

            # 发送控制指令到机械臂
            if ser:
                send_control_command(ser, control_x, control_y, control_z)
        
        # 绘制控制信息
        draw_control_info(frame, control_x, control_y, control_z)
        
        # 更新图表
        if enable_plot:
            update_plot(time_values, pid_x_values, pid_y_values, pid_z_values, 
                       line_x, line_y, line_z, ax, current_time, start_time, 
                       control_x, control_y, control_z)
        
        # 绘制图像中心点
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        
        # 显示结果帧
        cv2.imshow('Robot niu_ma Tracking', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    if ser:
        ser.close()

    # 关闭绘图窗口
    if enable_plot:
        plt.ioff()
        plt.close(fig)

if __name__ == "__main__":
    main()
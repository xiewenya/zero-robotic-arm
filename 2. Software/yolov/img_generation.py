import cv2
import os

# 创建image目录（如果不存在）
image_dir = "./image"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# 图片计数器
img_counter = 0

print("按 ENTER 键保存图片")
print("按 ESC 键退出程序")

while True:
    # 读取帧
    ret, frame = cap.read()
    
    if not ret:
        print("无法接收帧")
        break
    
    # 显示帧
    cv2.imshow('Camera', frame)
    
    # 等待按键输入
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # 按ESC键退出
        print("退出程序")
        break
    elif key == 13: # 按回车键保存图片
        img_name = os.path.join(image_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} 保存成功!".format(img_name))
        img_counter += 1

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
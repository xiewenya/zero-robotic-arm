import cv2
import os
import glob
import argparse

def resize_and_pad_images(input_folder="./image", output_folder="./processed_images", target_size=(640, 640)):
    """
    将指定文件夹中的所有图片调整为指定大小，使用黑色填充以保持宽高比
    
    参数:
    input_folder: 输入图片文件夹路径
    output_folder: 输出图片文件夹路径
    target_size: 目标尺寸 (宽, 高)
    """
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # 获取所有图片文件
    image_files = []
    for extension in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    if not image_files:
        print(f"在 {input_folder} 文件夹中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片文件
    for i, image_path in enumerate(image_files):
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"无法读取图片: {image_path}")
                continue
            
            # 获取原始尺寸
            original_height, original_width = img.shape[:2]
            
            # 计算缩放比例，保持宽高比
            scale = min(target_size[0] / original_width, target_size[1] / original_height)
            
            # 计算新尺寸
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # 调整图片大小
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 创建黑色背景图片
            padded_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            padded_img[:] = (0, 0, 0)  # 黑色填充
            
            # 计算在黑色背景上的位置（居中）
            x_offset = (target_size[0] - new_width) // 2
            y_offset = (target_size[1] - new_height) // 2
            
            # 将调整大小后的图片放在黑色背景上
            padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
            
            # 生成输出文件路径
            filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, filename)
            
            # 保存处理后的图片
            cv2.imwrite(output_path, padded_img)
            print(f"已处理 {i+1}/{len(image_files)}: {output_path}")
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {str(e)}")
    
    print(f"所有图片已处理完成，结果保存在 {output_folder} 文件夹中")

def main():
    parser = argparse.ArgumentParser(description='将图片调整为指定大小并用黑色填充')
    parser.add_argument('-i', '--input', default='./image', help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', default='./processed_images', help='输出图片文件夹路径')
    parser.add_argument('-s', '--size', nargs=2, type=int, default=[640, 640], 
                        help='目标尺寸，格式为 宽 高，例如：-s 640 640')
    
    args = parser.parse_args()
    
    resize_and_pad_images(
        input_folder=args.input,
        output_folder=args.output,
        target_size=tuple(args.size)
    )

if __name__ == "__main__":
    main()
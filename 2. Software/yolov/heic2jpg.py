import os
import sys
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(input_path, output_path=None):
    """
    将单个HEIC文件转换为JPG格式
    
    Args:
        input_path (str): 输入HEIC文件路径
        output_path (str): 输出JPG文件路径，默认为None，表示在相同位置生成JPG文件
    
    Returns:
        bool: 转换成功返回True，否则返回False
    """
    try:
        # 如果没有指定输出路径，则在相同目录下生成同名的JPG文件
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '.jpg'
        
        # 使用pillow-heif读取HEIC文件
        heif_file = pillow_heif.read_heif(input_path)
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
        )
        
        # 转换为RGB模式（如果需要）
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # 保存为JPG格式
        image.save(output_path, "JPEG")
        print(f"成功转换: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"转换失败: {input_path} - {str(e)}")
        return False

def batch_convert_heic_to_jpg(input_dir, output_dir=None):
    """
    批量将目录中的HEIC文件转换为JPG格式
    
    Args:
        input_dir (str): 包含HEIC文件的输入目录
        output_dir (str): 输出JPG文件的目录，默认为None，表示在相同目录下生成JPG文件
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        return
    
    # 如果没有指定输出目录，则使用输入目录
    if output_dir is None:
        output_dir = input_dir
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 统计转换结果
    success_count = 0
    fail_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查是否为HEIC文件（不区分大小写）
        if filename.lower().endswith('.heic'):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            # 转换单个文件
            if convert_heic_to_jpg(input_path, output_path):
                success_count += 1
            else:
                fail_count += 1
    
    # 输出转换结果统计
    print(f"\n转换完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir}")

def main():
    """
    主函数，处理命令行参数
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python heic2jpg.py <输入目录> [输出目录]")
        print("  python heic2jpg.py <输入文件> [输出文件]")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 判断输入是文件还是目录
    if os.path.isfile(input_path):
        # 处理单个文件
        if not input_path.lower().endswith('.heic'):
            print("错误: 输入文件不是HEIC格式")
            return
        
        if output_path and os.path.isdir(output_path):
            # 如果输出路径是目录，则在该目录下生成同名JPG文件
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + '.jpg'
            output_path = os.path.join(output_path, output_filename)
        
        convert_heic_to_jpg(input_path, output_path)
    elif os.path.isdir(input_path):
        # 处理整个目录
        batch_convert_heic_to_jpg(input_path, output_path)
    else:
        print(f"错误: 输入路径 {input_path} 不存在")

if __name__ == "__main__":
    main()
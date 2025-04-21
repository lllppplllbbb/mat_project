import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_masks(img_dir):
    """
    测试掩码图片是否符合要求：
    1. .png格式
    2. 128*128分辨率
    3. 像素值为0或255
    """
    mask_dir = os.path.join(img_dir, "masks")
    if not os.path.exists(mask_dir):
        print(f"错误：掩码目录 {mask_dir} 不存在！")
        return
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    if not mask_files:
        print(f"错误：在 {mask_dir} 中没有找到.png格式的掩码文件！")
        return
    
    print(f"找到 {len(mask_files)} 个掩码文件，开始测试...")
    
    all_valid = True
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        try:
            # 读取掩码图片
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            
            # 检查分辨率
            if mask_array.shape != (128, 128):
                print(f"错误：{mask_file} 分辨率不是128*128，实际为 {mask_array.shape}")
                all_valid = False
                continue
            
            # 检查像素值
            unique_values = np.unique(mask_array)
            if not np.array_equal(unique_values, np.array([0, 255])) and not np.array_equal(unique_values, np.array([0])) and not np.array_equal(unique_values, np.array([255])):
                print(f"错误：{mask_file} 像素值不只是0和255，实际值有 {unique_values}")
                all_valid = False
                continue
            
            # 显示进度
            if (i+1) % 10 == 0:
                print(f"已测试 {i+1}/{len(mask_files)} 个掩码文件")
                
        except Exception as e:
            print(f"错误：处理 {mask_file} 时发生异常：{str(e)}")
            all_valid = False
    
    if all_valid:
        print("测试通过！所有掩码文件都符合要求：.png格式，128*128分辨率，像素值为0或255")
    else:
        print("测试失败！部分掩码文件不符合要求")
    
    # 可视化几个随机掩码
    visualize_random_masks(mask_dir, mask_files, num=4)

def visualize_random_masks(mask_dir, mask_files, num=4):
    """可视化几个随机掩码"""
    if len(mask_files) < num:
        num = len(mask_files)
    
    plt.figure(figsize=(10, 10))
    for i in range(num):
        random_idx = np.random.randint(0, len(mask_files))
        mask_path = os.path.join(mask_dir, mask_files[random_idx])
        mask = Image.open(mask_path)
        
        plt.subplot(2, 2, i+1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"{mask_files[random_idx]}\n{np.array(mask).shape}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(mask_dir), "mask_samples.png"))
    plt.close()
    print(f"已保存随机掩码样本图片到 {os.path.join(os.path.dirname(mask_dir), 'mask_samples.png')}")

if __name__ == "__main__":
    # 测试训练图像的掩码
    train_img_dir = "F:/MAT_project/MAT/data/train_images"
    test_img_dir = "F:/MAT_project/MAT/data/test_images"
    
    print("测试训练图像掩码...")
    test_masks(train_img_dir)
    
    print("\n测试验证图像掩码...")
    test_masks(test_img_dir)
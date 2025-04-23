import torch
import platform
import sys
import os
import cv2
import numpy as np
from datasets.dataset_512_val import ImageFolderMaskDataset

# 目录路径设置
train_images_dir = 'F:/MAT_project/MAT/data/train_images'
train_seg_dir = 'F:/MAT_project/MAT/data/train_images/train_seg'
mask_dir = 'F:/MAT_project/MAT/data/train_images/masks'

# 检查目录是否存在
print("检查目录状态：")
print(f"训练图像目录是否存在: {os.path.exists(train_images_dir)}")
print(f"分割图目录是否存在: {os.path.exists(train_seg_dir)}")
print(f"掩码目录是否存在: {os.path.exists(mask_dir)}")

# # CUDA环境检查
# print("\nCUDA环境信息：")
# print(f"Python版本: {platform.python_version()}")
# print(f"PyTorch版本: {torch.__version__}")
# print(f"CUDA是否可用: {torch.cuda.is_available()}")

# if torch.cuda.is_available():
#     print(f"CUDA版本: {torch.version.cuda}")
#     print(f"cuDNN版本: {torch.backends.cudnn.version()}")
#     print(f"当前CUDA设备: {torch.cuda.current_device()}")
#     print(f"CUDA设备数量: {torch.cuda.device_count()}")
#     print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA设备属性: {torch.cuda.get_device_properties(0)}")

#     # 测试CUDA张量运算
#     try:
#         x = torch.rand(5, 3).cuda()
#         y = torch.rand(5, 3).cuda()
#         z = x + y
#         print("CUDA张量运算测试成功")
#     except Exception as e:
#         print(f"CUDA张量运算测试失败: {e}")

# 测试分割图加载
try:
    seg_path = os.path.join(train_seg_dir, '2008_000015.png')  # 示例分割图路径
    if os.path.exists(seg_path):
        seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        print(f"\n分割图信息:")
        print(f"形状: {seg.shape}")
        print(f"最小值: {seg.min()}")
        print(f"最大值: {seg.max()}")
    else:
        print(f"\n警告: 示例分割图 {seg_path} 不存在")
except Exception as e:
    print(f"分割图加载失败: {e}")

# 测试数据集加载
try:
    dataset = ImageFolderMaskDataset(
        path=train_images_dir,
        resolution=512,
        seg_dir=train_seg_dir
    )
    print(f"\n数据集信息:")
    print(f"图像数量: {len(dataset)}")
    if hasattr(dataset, 'masks'):
        print(f"掩码数量: {len(dataset.masks)}")
    if hasattr(dataset, 'segs'):
        print(f"分割图数量: {len(dataset.segs)}")
except Exception as e:
    print(f"数据集加载失败: {e}")
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets.dataset_512 import ImageFolderMaskDataset as TrainDataset
from datasets.dataset_512_val import ImageFolderMaskDataset as ValDataset
from losses.loss import semantic_loss

# Set paths - adjust these to your environment
base = 'F:/MAT_project/MAT/data'  # Local path for testing
# base = '/kaggle/input/pascal-voc-mat/MAT_images_20250423'  # Kaggle path for deployment

# 加载分割图目录
train_seg_dir = f'{base}/segmentations/train'
test_seg_dir = f'{base}/segmentations/test'

# 检查分割图目录是否存在
print(f"检查分割图目录:")
print(f"训练集分割图目录存在: {os.path.exists(train_seg_dir)}")
print(f"测试集分割图目录存在: {os.path.exists(test_seg_dir)}")

# Load datasets
train_set = TrainDataset(f'{base}/train_images', resolution=512, seg_dir=train_seg_dir)
val_set = ValDataset(f'{base}/test_images', resolution=512, seg_dir=test_seg_dir)

# Test dataset loading
print("\n测试数据集加载...")
try:
    # 检查返回值的数量
    result = train_set[0]
    if len(result) == 4:
        img, mask, seg, label = result
        print(f"训练集 - 图像: {img.shape}, 掩码: {mask.shape}, 分割图: {seg.shape}")
    elif len(result) == 3:
        img, mask, label = result
        print(f"训练集 - 图像: {img.shape}, 掩码: {mask.shape}")
        print("警告: 没有找到分割图，请确保分割图目录正确并包含图像")
        # 创建一个假的分割图用于测试
        seg = np.ones((1, img.shape[1], img.shape[2]), dtype=np.float32)
    else:
        raise ValueError(f"意外的返回值数量: {len(result)}")
except Exception as e:
    print(f"训练集加载失败: {e}")
    # 创建测试用的假数据
    img = np.random.randint(0, 255, (3, 512, 512), dtype=np.uint8)
    mask = np.ones((1, 512, 512), dtype=np.float32)
    seg = np.ones((1, 512, 512), dtype=np.float32)
    label = np.zeros(1, dtype=np.float32)

try:
    # 检查返回值的数量
    result = val_set[0]
    if len(result) == 4:
        val_img, val_mask, val_seg, val_label = result
        print(f"验证集 - 图像: {val_img.shape}, 掩码: {val_mask.shape}, 分割图: {val_seg.shape}")
    elif len(result) == 3:
        val_img, val_mask, val_label = result
        print(f"验证集 - 图像: {val_img.shape}, 掩码: {val_mask.shape}")
        print("警告: 没有找到分割图，请确保分割图目录正确并包含图像")
    else:
        raise ValueError(f"意外的返回值数量: {len(result)}")
except Exception as e:
    print(f"验证集加载失败: {e}")

# Test semantic loss function
print("\n测试语义损失函数...")
# Convert numpy arrays to PyTorch tensors and ensure they are float type
img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # 转换为float类型并添加批次维度
seg_tensor = torch.from_numpy(seg).float().unsqueeze(0)  # 确保分割图也是float类型

# Create a "prediction" by adding some noise to the original image
pred_tensor = img_tensor + torch.randn_like(img_tensor) * 0.1
pred_tensor = torch.clamp(pred_tensor, 0, 255)

# Calculate semantic loss
try:
    loss = semantic_loss(pred_tensor, img_tensor, seg_tensor)
    print(f"语义损失值: {loss.item()}")
except Exception as e:
    print(f"计算语义损失失败: {e}")

# Visualize the results
def visualize_results():
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(np.transpose(img, (1, 2, 0)).astype(np.uint8))
    plt.title("原始图像")
    plt.axis('off')
    
    # Mask
    plt.subplot(2, 3, 2)
    plt.imshow(mask[0], cmap='gray')
    plt.title("掩码")
    plt.axis('off')
    
    # Segmentation
    plt.subplot(2, 3, 3)
    plt.imshow(seg[0], cmap='jet')
    plt.title("分割图")
    plt.axis('off')
    
    # Prediction
    plt.subplot(2, 3, 4)
    pred_np = pred_tensor.squeeze(0).numpy().astype(np.uint8)  # 确保转换回uint8类型用于显示
    plt.imshow(np.transpose(pred_np, (1, 2, 0)))
    plt.title("预测 (添加噪声)")
    plt.axis('off')
    
    # Weighted by segmentation
    plt.subplot(2, 3, 5)
    weighted_img = img * seg
    plt.imshow(np.transpose(weighted_img, (1, 2, 0)).astype(np.uint8))
    plt.title("分割图加权图像")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{base}/semantic_loss_test.png")
    print(f"可视化结果已保存到 {base}/semantic_loss_test.png")

try:
    visualize_results()
except Exception as e:
    print(f"可视化失败: {e}")

print("测试完成!")
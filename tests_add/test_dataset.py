import os
import sys
import numpy as np
import cv2

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.dataset_512_val import ImageFolderMaskDataset

train_img_path = 'F:/MAT_project/MAT/data/train_images'
train_seg_path = 'F:/MAT_project/MAT/data/segmentations/train'

print("正在加载数据集...")
D = ImageFolderMaskDataset(path=train_img_path, seg_dir=train_seg_path)
print(f"数据集大小: {len(D)}")

all_classes = set()
class_counts = {}
for i in range(len(D)):
    img, mask, seg, label = D[i]
    unique_classes = np.unique(seg)
    unique_mask = np.unique(mask)
    all_classes.update(unique_classes)
    for cls in unique_classes:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    if i < 50:
        print(f"样本 {i} - 图像形状: {img.shape}, 掩码形状: {mask.shape}, 分割图形状: {seg.shape}, 标签形状: {label.shape}")
        print(f"样本 {i} - 分割图唯一值: {unique_classes}, 掩码值: {unique_mask}, 掩码类型: {mask.dtype}")
    if not np.all(np.isin(unique_mask, [0, 1])):
        print(f"[WARNING] 样本 {i} 掩码值非二值: {unique_mask}")
    
    # 获取掩码文件路径
    img_filename = os.path.basename(D._image_fnames[D._raw_idx[i]])
    mask_filename = os.path.splitext(img_filename)[0] + '.png'
    mask_path = os.path.join(train_img_path, 'masks', mask_filename)
    
    # 打印原始掩码的唯一值
    if os.path.exists(mask_path):
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            print(f"原始掩码: {np.unique(raw_mask)}")
        else:
            print(f"无法读取掩码文件: {mask_path}")
    else:
        print(f"掩码文件不存在: {mask_path}")

print(f"所有样本的类别集合: {sorted(all_classes)}")
print(f"类别分布: {class_counts}")

# 安全地打印transform属性
if hasattr(D, 'transform'):
    print(f"变换: {D.transform}")
else:
    print("数据集对象没有transform属性")
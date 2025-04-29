import os
import shutil
import numpy as np
from PIL import Image

voc_root = "F:/MAT_project/data/VOC2012/VOCdevkit/VOC2012"
train_img_dir = "F:/MAT_project/MAT/data/train_images"
train_seg_dir = "F:/MAT_project/MAT/data/segmentations/train"  # 修正目录
test_img_dir = "F:/MAT_project/MAT/data/test_images"
test_seg_dir = "F:/MAT_project/MAT/data/segmentations/test"   # 修正目录

def clear_directory(directory):
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"已清空目录: {directory}")

# 清空目标目录
clear_directory(train_img_dir)
clear_directory(train_seg_dir)
clear_directory(test_img_dir)
clear_directory(test_seg_dir)

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_seg_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(test_seg_dir, exist_ok=True)

# 读训练和验证列表
with open(os.path.join(voc_root, "ImageSets/Main/train.txt"), "r") as f:
    train_list = f.read().splitlines()
with open(os.path.join(voc_root, "ImageSets/Main/val.txt"), "r") as f:
    val_list = f.read().splitlines()

# 统计类别分布
def get_seg_classes(seg_path):
    seg = np.array(Image.open(seg_path))
    seg[seg == 255] = 0  
    return np.unique(seg)

# 选择前景丰富的训练样本
train_copied = 0
train_class_counts = {}
selected_train = []
for name in train_list:
    img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
    seg_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")
    if os.path.exists(img_path) and os.path.exists(seg_path):
        classes = get_seg_classes(seg_path)
        if len(classes) > 1 or (len(classes) == 1 and classes[0] != 0):  # 包含前景
            selected_train.append(name)
            for cls in classes:
                train_class_counts[cls] = train_class_counts.get(cls, 0) + 1
    if len(selected_train) >= 200:
        break

# 复制训练图
for name in selected_train[:200]:
    img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
    seg_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")
    seg = Image.open(seg_path)
    seg_np = np.array(seg)
    seg_np[seg_np == 255] = 0
    Image.fromarray(seg_np).save(os.path.join(train_seg_dir, f"{name}.png"))
    shutil.copy(img_path, train_img_dir)
    train_copied += 1
    print(f"已复制训练图片 {train_copied}/200: {name}")
print(f"训练集类别分布: {train_class_counts}")

# 复制测试图
test_copied = 0
test_class_counts = {}
for name in val_list:
    if test_copied >= 50:
        break
    img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
    seg_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")
    if os.path.exists(img_path) and os.path.exists(seg_path):
        classes = get_seg_classes(seg_path)
        shutil.copy(img_path, test_img_dir)
        shutil.copy(seg_path, test_seg_dir)
        test_copied += 1
        for cls in classes:
            test_class_counts[cls] = test_class_counts.get(cls, 0) + 1
        print(f"已复制测试图片及分割图 {test_copied}/50: {name}")
print(f"测试集类别分布: {test_class_counts}")

print(f"完成！已复制 {train_copied} 张训练图片和分割图，{test_copied} 张测试图片和分割图。")

def count_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"当前训练图片数量: {count_images(train_img_dir)}")
print(f"当前训练分割图数量: {count_images(train_seg_dir)}")
print(f"当前测试图片数量: {count_images(test_img_dir)}")
print(f"当前测试分割图数量: {count_images(test_seg_dir)}")
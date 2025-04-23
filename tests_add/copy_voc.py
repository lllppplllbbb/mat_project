import os
import shutil

voc_root = "F:/MAT_project/data/VOC2012/VOCdevkit/VOC2012"
train_img_dir = "F:/MAT_project/MAT/data/train_images"
train_seg_dir = "F:/MAT_project/MAT/data/train_seg"
test_img_dir = "F:/MAT_project/MAT/data/test_images"
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_seg_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# 读训练列表
with open(os.path.join(voc_root, "ImageSets/Main/train.txt"), "r") as f:
    train_list = f.read().splitlines()
# 读验证列表
with open(os.path.join(voc_root, "ImageSets/Main/val.txt"), "r") as f:
    val_list = f.read().splitlines()

# 复制训练图和分割图，直到够200张
train_copied = 0
for name in train_list:
    if train_copied >= 200:
        break
    img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
    seg_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")
    if os.path.exists(img_path) and os.path.exists(seg_path):
        shutil.copy(img_path, train_img_dir)
        shutil.copy(seg_path, train_seg_dir)
        train_copied += 1
        print(f"已复制训练图片 {train_copied}/200: {name}")      #需变换数量
    else:
        print(f"跳过：{name}，因图片或分割标注不存在。")

# 复制测试图，直到够50张
test_copied = 0
for name in val_list:
    if test_copied >= 50:
        break
    img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
    if os.path.exists(img_path):
        shutil.copy(img_path, test_img_dir)
        test_copied += 1
        print(f"已复制测试图片 {test_copied}/50: {name}")       #需变换数量
    else:
        print(f"跳过：{name}，因图片不存在。")

print(f"完成！已复制 {train_copied} 张训练图片和分割图，{test_copied} 张测试图片。")
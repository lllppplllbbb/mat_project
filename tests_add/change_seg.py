from PIL import Image
import os
import shutil

def resize_segmentations(img_dir, seg_dir, out_dir, resolution=512):
    """调整分割图像大小并保存到输出目录"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 统计信息
    total_images = 0
    processed_images = 0
    missing_images = 0
    
    # 获取所有分割图文件名
    if os.path.exists(seg_dir):
        seg_files = os.listdir(seg_dir)
        print(f"在 {seg_dir} 中找到 {len(seg_files)} 个分割图文件")
    else:
        seg_files = []
        print(f"警告: 分割图目录 {seg_dir} 不存在")
    
    for seg_name in seg_files:
        if seg_name.endswith('.png'):
            total_images += 1
            seg_path = os.path.join(seg_dir, seg_name)
            
            try:
                seg = Image.open(seg_path)
                seg = seg.resize((resolution, resolution), Image.NEAREST)  # NEAREST保持类别值
                seg.save(os.path.join(out_dir, seg_name))
                processed_images += 1
                print(f"处理: {seg_name}")
            except Exception as e:
                missing_images += 1
                print(f"处理失败: {seg_name}, 错误: {e}")
    
    print(f"\n处理完成:")
    print(f"总分割图数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"处理失败: {missing_images}")
    
    return processed_images

# 目录设置 - 使用与copy_voc.py相同的路径结构
train_img_dir = "F:/MAT_project/MAT/data/train_images"
train_seg_dir = "F:/MAT_project/MAT/data/segmentations/original/train_seg"
train_out_dir = "F:/MAT_project/MAT/data/segmentations/train"

test_img_dir = "F:/MAT_project/MAT/data/test_images"
test_seg_dir = "F:/MAT_project/MAT/data/segmentations/original/test_seg"
test_out_dir = "F:/MAT_project/MAT/data/segmentations/test"

# 处理训练集分割图
print("处理训练集分割图...")
train_processed = resize_segmentations(
    img_dir=train_img_dir,
    seg_dir=train_seg_dir,
    out_dir=train_out_dir,
    resolution=512
)

# 处理测试集分割图
print("\n处理测试集分割图...")
test_processed = resize_segmentations(
    img_dir=test_img_dir,
    seg_dir=test_seg_dir,
    out_dir=test_out_dir,
    resolution=512
)

print("\n所有处理完成!")
print(f"训练集: 处理了 {train_processed} 张分割图到 {train_out_dir}")
print(f"测试集: 处理了 {test_processed} 张分割图到 {test_out_dir}")
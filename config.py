import os

# 检测是否在Kaggle环境中
def is_kaggle():
    return os.path.exists('/kaggle/input')

# 根据环境设置路径
if is_kaggle():
    # Kaggle环境路径
    DATA_PATHS = {
        'train_data': "/kaggle/input/pascal-voc-mat/MAT_images_20250423/train_images",
        'train_mask': "/kaggle/input/pascal-voc-mat/MAT_images_20250423/train_images/masks",
        'val_data': "/kaggle/input/pascal-voc-mat/MAT_images_20250423/test_images",
        'val_mask': "/kaggle/input/pascal-voc-mat/MAT_images_20250423/test_images/masks",
    }
    DEBUG = True
else:
    # 本地环境路径
    DATA_PATHS = {
        'train_data': "F:/MAT_project/MAT/data/train_images",
        'train_mask': "F:/MAT_project/MAT/data/train_images/masks",
        'val_data': "F:/MAT_project/MAT/data/test_images",
        'val_mask': "F:/MAT_project/MAT/data/test_images/masks",
    }
    DEBUG = False
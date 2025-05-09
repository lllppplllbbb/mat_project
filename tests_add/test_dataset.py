from datasets.dataset_512 import ImageFolderMaskDataset
import numpy as np
import torch

def test_dataset():
    dataset = ImageFolderMaskDataset(
        path='F:/MAT_project/MAT/data/train_images',
        resolution=512,
        max_size=10,
        seg_dir='F:/MAT_project/MAT/data/segmentations/train'
    )
    for i, (img, mask, seg, label) in enumerate(dataset):
        # 统一转换为 PyTorch 张量
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        if isinstance(seg, np.ndarray):
            seg = torch.from_numpy(seg).float()
        
        # 确保数据类型和形状
        img = img.to(torch.float32)
        mask = mask.to(torch.float32)
        seg = seg.to(torch.float32)
        
        print(f"[DEBUG] 样本 {i} - 图像形状: {img.shape}, 掩码形状: {mask.shape}, 分割图形状: {seg.shape}")
        print(f"[DEBUG] 掩码唯一值: {torch.unique(mask).numpy()}")
        print(f"[DEBUG] 分割图唯一值: {torch.unique(seg).numpy()}")
        foreground_pixels = (seg > 0).sum().item()
        total_pixels = seg.numel()
        print(f"[DEBUG] 前景像素数: {foreground_pixels}, 占比: {foreground_pixels/total_pixels:.2%}")
        if i >= 4:
            break

if __name__ == "__main__":
    test_dataset()
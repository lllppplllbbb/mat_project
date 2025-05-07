import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.loss import TwoStageLoss, generate_weight_map, semantic_loss
from torch_utils import misc
from datasets.dataset_512 import ImageFolderMaskDataset  # 修改导入为dataset_512.py

dataset = ImageFolderMaskDataset(path='F:/MAT_project/MAT/data/train_images', 
                                seg_dir='F:/MAT_project/MAT/data/segmentations/train', 
                                mask_dir='F:/MAT_project/MAT/data/train_images/masks',  # 显式指定mask_dir
                                resolution=512)

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x

def test_perceptual_loss():
    print("测试语义加权感知损失...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    G_mapping = DummyModel().to(device)
    G_synthesis = DummyModel().to(device)
    D = DummyModel().to(device)

    loss_obj = TwoStageLoss(
        device=device, 
        G_mapping=G_mapping, 
        G_synthesis=G_synthesis, 
        D=D,
        pcp_ratio=0.5,
        sem_ratio=0.3
    )

    batch_size = 2
    resolution = 512  # 统一为512
    pred = torch.randn(batch_size, 3, resolution, resolution, device=device)
    target = torch.randn(batch_size, 3, resolution, resolution, device=device)
    seg = torch.randint(0, 20, (batch_size, 1, resolution, resolution), device=device).float()  # 类别数统一为20

    weight_map = generate_weight_map(seg)
    print(f"权重图形状: {weight_map.shape}")
    print(f"权重图唯一值: {torch.unique(weight_map).cpu().numpy()}")

    sem_loss = semantic_loss(pred, target, seg)
    print(f"语义损失: {sem_loss.item():.4f}")

    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, seg)
    print(f"加权感知损失: {pcp_loss.item():.4f}")

    print("\n测试不同类别比例的分割图:")
    bg_seg = torch.zeros(batch_size, 1, resolution, resolution, device=device).float()
    mask = torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.9  # 10%掩码
    bg_seg[mask] = torch.randint(1, 20, (mask.sum().item(),), device=device).float()
    bg_weights = generate_weight_map(bg_seg)
    bg_pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, bg_seg)
    print(f"背景为主的加权感知损失: {bg_pcp_loss.item():.4f}")

    fg_seg = torch.randint(1, 20, (batch_size, 1, resolution, resolution), device=device).float()
    mask = torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.9
    fg_seg[mask] = 0
    fg_pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, fg_seg)
    print(f"前景为主的加权感知损失: {fg_pcp_loss.item():.4f}")

    print("\n测试真实样本:")
    for i in range(min(10, len(dataset))):
        img, mask, seg, _ = dataset[i]
        img = torch.from_numpy(img).to(device).unsqueeze(0) / 127.5 - 1.0  # 归一化到[-1, 1]
        mask = torch.from_numpy(mask).to(device).unsqueeze(0).float()
        seg = torch.from_numpy(seg).to(device).unsqueeze(0).float()
        pred = torch.randn(1, 3, 512, 512, device=device)  # 替换为真实生成器输出（如果有）
        pcp_loss = loss_obj.perceptual_loss_with_weights(pred, img, seg)
        mask_ratio = mask.mean().item()  # 计算掩码面积比例
        print(f"样本 {i} - 加权感知损失: {pcp_loss.item():.4f}, 掩码面积比例: {mask_ratio:.4f}")
        print(f"样本 {i} - 分割图类别: {np.unique(seg.cpu().numpy())}")

    print("\n测试完成!")

if __name__ == "__main__":
    test_perceptual_loss()
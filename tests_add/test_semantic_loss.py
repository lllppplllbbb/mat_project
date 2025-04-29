import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses.loss import TwoStageLoss, generate_weight_map, semantic_loss
from torch_utils import misc
from datasets.dataset_512 import Dataset512
dataset = Dataset512(image_dir='F:/MAT_project/MAT/data/train_images', seg_dir='F:/MAT_project/MAT/data/segmentations/train')
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
    resolution = 256
    pred = torch.randn(batch_size, 3, resolution, resolution, device=device)
    target = torch.randn(batch_size, 3, resolution, resolution, device=device)
    seg = torch.randint(0, 21, (batch_size, 1, resolution, resolution), device=device).float()
    
    weight_map = generate_weight_map(seg)
    print(f"权重图形状: {weight_map.shape}")
    print(f"权重图唯一值: {torch.unique(weight_map).cpu().numpy()}")
    
    sem_loss = semantic_loss(pred, target, seg)
    print(f"语义损失: {sem_loss.item():.4f}")
    
    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, seg)
    print(f"加权感知损失: {pcp_loss.item():.4f}")
    
    print("\n测试不同类别比例的分割图:")
    bg_seg = torch.zeros(batch_size, 1, resolution, resolution, device=device).float()
    mask = torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.9
    bg_seg[mask] = torch.randint(1, 21, (mask.sum().item(),), device=device).float()
    bg_weights = generate_weight_map(bg_seg)
    bg_pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, bg_seg)
    print(f"背景为主的加权感知损失: {bg_pcp_loss.item():.4f}")
    
    fg_seg = torch.randint(1, 21, (batch_size, 1, resolution, resolution), device=device).float()
    mask = torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.9
    fg_seg[mask] = 0
    fg_pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, fg_seg)
    print(f"前景为主的加权感知损失: {fg_pcp_loss.item():.4f}")

    for i in range(10):
        img, _, seg, _ = dataset[i]
        pred = torch.randn(1, 3, 512, 512, device=device)
        target = torch.tensor(img, device=device).unsqueeze(0)
        seg = torch.tensor(seg, device=device).unsqueeze(0)
        pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, seg)
        print(f"样本 {i} - 加权感知损失: {pcp_loss.item():.4f}")
    # print("\n测试多个随机样本:")
    # for i in range(10):  # 测试10张
    #     pred = torch.randn(2, 3, 512, 512, device=device)
    #     target = torch.randn(2, 3, 512, 512, device=device)
    #     seg = torch.randint(0, 21, (2, 1, 256, 256), device=device).float()
    #     pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, seg)
    #     print(f"样本 {i} - 加权感知损失: {pcp_loss.item():.4f}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_perceptual_loss()
import os
import sys
import torch
import numpy as np
from losses.loss import TwoStageLoss, generate_weight_map, semantic_loss
from torch_utils import misc

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x

def test_loss_balance():
    print("测试损失平衡情况...")
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
        sem_ratio=1.0
    )
    
    batch_size = 2
    resolution = 256
    pred = torch.randn(batch_size, 3, resolution, resolution, device=device)
    target = torch.randn(batch_size, 3, resolution, resolution, device=device)
    mask = torch.rand(batch_size, 1, resolution, resolution, device=device) > 0.7
    seg = torch.randint(0, 21, (batch_size, 1, resolution, resolution), device=device).float()
    
    adv_loss = torch.nn.functional.mse_loss(pred, target) * 1.2
    print(f"对抗损失  {adv_loss.item():.4f}")
    
    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, target, seg)
    print(f"感知损失 (原始): {pcp_loss.item():.4f}")
    print(f"感知损失 : {(pcp_loss * 0.6).item():.4f}")
    
    sem_loss = semantic_loss(pred, target, seg)
    print(f"语义损失 (原始): {sem_loss.item():.4f}")
    print(f"语义损失 : {(sem_loss * 0.4).item():.4f}")
    
    total_loss = adv_loss + pcp_loss * 0.6 + sem_loss * 0.4
    print(f"总损失: {total_loss.item():.4f}")
    
    adv_ratio = adv_loss.item() / total_loss.item()
    pcp_ratio = (pcp_loss.item() * 0.6) / total_loss.item()
    sem_ratio = (sem_loss.item() * 0.4) / total_loss.item()
    
    print("\n损失比例分析:")
    print(f"对抗损失占比: {adv_ratio:.2%}")
    print(f"感知损失占比: {pcp_ratio:.2%}")
    print(f"语义损失占比: {sem_ratio:.2%}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_loss_balance()
import torch
import numpy as np
from losses.loss import TwoStageLoss, generate_weight_map, semantic_loss
from datasets.dataset_512 import ImageFolderMaskDataset
from torch_utils import misc
import os
import torchvision.utils as vutils

class DummyModel(torch.nn.Module):
    def __init__(self, is_discriminator=False):
        super().__init__()
        self.is_discriminator = is_discriminator
        # 生成器和判别器：更深网络，保留输入结构
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 输入：img (3) + mask (1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(64)
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)

    def forward(self, *args, **kwargs):
        if len(args) > 0:
            input_tensor = args[0].float() if args[0].dtype != torch.float32 else args[0]
            mask = args[1].float() if len(args) > 1 else torch.zeros_like(input_tensor[:, :1, :, :])
            input_tensor = torch.cat([input_tensor, mask], dim=1)  # 拼接 img 和 mask
            x = self.relu(self.bn1(self.conv1(input_tensor)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.conv4(x)
            if self.is_discriminator:
                # 判别器：全局平均池化
                logits = x.mean(dim=(2, 3))
                return logits, logits * 0.5  # 模拟两阶段 logits
            # 生成器：融合输入图像和掩码
            noise = torch.randn_like(x) * 0.005  # 降低噪声强度
            output = torch.tanh(x + noise)
            # 保留非掩码区域的原始图像
            output = input_tensor[:, :3, :, :] * (1 - mask) + output * mask
            return output
        else:
            if self.is_discriminator:
                return torch.randn(1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), torch.randn(1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            return torch.tanh(torch.randn(1, 3, 512, 512, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

def test_loss_balance():
    print("测试损失平衡情况...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载小规模数据集（512x512）
    dataset = ImageFolderMaskDataset(
        path='F:/MAT_project/MAT/data/train_images',
        resolution=512,
        max_size=10,
        seg_dir='F:/MAT_project/MAT/data/segmentations/train'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 初始化模型和损失函数
    G_mapping = DummyModel().to(device)
    G_synthesis = DummyModel().to(device)
    D = DummyModel(is_discriminator=True).to(device)
    loss_obj = TwoStageLoss(device=device, G_mapping=G_mapping, G_synthesis=G_synthesis, D=D, pcp_ratio=0.5, sem_ratio=0.3)
    
    # 测试损失
    os.makedirs('MAT/test_results', exist_ok=True)
    for i, (img, mask, seg, _) in enumerate(dataloader):
        img, mask, seg = img.to(device), mask.to(device), seg.to(device)
        gen_img = loss_obj.G_synthesis(img * (1 - mask), mask)
        gen_logits, gen_logits_stg1 = loss_obj.D(gen_img, mask)
        adv_loss = (torch.nn.functional.softplus(-gen_logits) + torch.nn.functional.softplus(-gen_logits_stg1)).mean() * 1000.0
        pcp_loss = loss_obj.perceptual_loss_with_weights(gen_img, img, seg)
        sem_loss = semantic_loss(gen_img, img, seg)
        
        # 计算总损失和占比
        total_loss = (loss_obj.weights['adv'] * adv_loss + 
                      loss_obj.weights['pcp'] * pcp_loss + 
                      loss_obj.weights['sem'] * sem_loss)
        total_value = total_loss.item()
        adv_ratio = (adv_loss.item() * loss_obj.weights['adv']) / total_value if total_value > 0 else 0
        pcp_ratio = (pcp_loss.item() * loss_obj.weights['pcp']) / total_value if total_value > 0 else 0
        sem_ratio = (sem_loss.item() * loss_obj.weights['sem']) / total_value if total_value > 0 else 0
        
        print(f"[TEST] 对抗损失: {adv_loss.item():.4f}, 占比: {adv_ratio:.2%} (目标: 50%)")
        print(f"[TEST] 感知损失: {pcp_loss.item():.4f}, 占比: {pcp_ratio:.2%} (目标: 25%)")
        print(f"[TEST] 语义损失: {sem_loss.item():.4f}, 占比: {sem_ratio:.2%} (目标: 25%)")
        print(f"[TEST] 总损失: {total_value:.4f}")
        print("-" * 50)
            
        # 保存生成图像、输入图像和掩码
        vutils.save_image(gen_img.clamp(-1, 1), f'MAT/test_results/pred_{i}.png', normalize=True, value_range=(-1, 1))
        vutils.save_image(img.float(), f'MAT/test_results/input_{i}.png', normalize=True, value_range=(-1, 1))
        vutils.save_image(mask.float(), f'MAT/test_results/mask_{i}.png', normalize=True, value_range=(0, 1))

if __name__ == "__main__":
    test_loss_balance()
import torch
import numpy as np
from losses.loss import TwoStageLoss, generate_weight_map, semantic_loss
from datasets.dataset_512 import ImageFolderMaskDataset
from torch_utils import misc
import os
import torchvision.utils as vutils

class DummyMapping(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 简单 MLP 将潜在向量 z 和条件 c 映射到 ws
        self.fc1 = torch.nn.Linear(512 + 1, 512)  # z: 512, c: 1
        self.fc2 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU(inplace=True)
        # 初始化权重
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, z, c, truncation_psi=1.0, **kwargs):
        # z: [batch, 512], c: [batch]
        c = c.view(-1, 1).float()  # 扩展 c 为 [batch, 1]
        x = torch.cat([z, c], dim=1)  # [batch, 513]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # [batch, 512]
        # 应用 truncation_psi
        if truncation_psi < 1:
            x = x * truncation_psi + torch.zeros_like(x) * (1 - truncation_psi)
        return x

class DummyModel(torch.nn.Module):
    def __init__(self, is_discriminator=False):
        super().__init__()
        self.is_discriminator = is_discriminator
        # 生成器和判别器：6 层卷积
        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 输入：img (3) + mask (1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(64)
        # 判别器线性层
        if is_discriminator:
            self.fc_logits = torch.nn.Linear(3, 1)  # 最终输出通道 3 -> 1
            self.fc_logits_stg1 = torch.nn.Linear(128, 1)  # 中间特征通道 128 -> 1
            torch.nn.init.xavier_uniform_(self.fc_logits.weight)
            torch.nn.init.xavier_uniform_(self.fc_logits_stg1.weight)
        # 初始化卷积权重
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, img_in, mask, ws=None, return_stg1=False):
        input_tensor = img_in.float() if img_in.dtype != torch.float32 else img_in
        mask = mask.float() if mask.dtype != torch.float32 else mask
        input_tensor = torch.cat([input_tensor, mask], dim=1)  # 拼接 img 和 mask
        x = self.relu(self.bn1(self.conv1(input_tensor)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_stg1 = x  # 保存第一阶段输出
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        if self.is_discriminator:
            # 判别器：全局平均池化 + 线性层
            logits = x.mean(dim=(2, 3))  # [batch, 3]
            logits = self.fc_logits(logits)  # [batch, 1]
            logits_stg1 = x_stg1.mean(dim=(2, 3))  # [batch, 128]
            logits_stg1 = self.fc_logits_stg1(logits_stg1)  # [batch, 1]
            return logits, logits_stg1
        # 生成器：融合输入图像和掩码
        noise = torch.randn_like(x) * 0.002  # 降低噪声强度
        output = torch.tanh(x + noise)
        # 保留非掩码区域的原始图像
        output = input_tensor[:, :3, :, :] * (1 - mask) + output * mask
        if return_stg1:
            return output, x_stg1
        return output

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
    G_mapping = DummyMapping().to(device)
    G_synthesis = DummyModel().to(device)
    D = DummyModel(is_discriminator=True).to(device)
    loss_obj = TwoStageLoss(device=device, G_mapping=G_mapping, G_synthesis=G_synthesis, D=D, pcp_ratio=0.5, sem_ratio=0.3)
    
    # 优化器
    optimizer_G = torch.optim.Adam(list(G_mapping.parameters()) + list(G_synthesis.parameters()), lr=0.001)
    
    # 测试损失
    os.makedirs('MAT/test_results', exist_ok=True)
    for i, (img, mask, seg, _) in enumerate(dataloader):
        # 确保输入张量为 float32
        img = img.to(device, dtype=torch.float32)
        mask = mask.to(device, dtype=torch.float32)
        seg = seg.to(device, dtype=torch.float32)
        gen_z = torch.randn(img.size(0), 512, device=device, dtype=torch.float32)
        gen_c = torch.arange(img.size(0), device=device, dtype=torch.float32)
        
        # 模拟优化步骤（10 次迭代）
        for _ in range(10):
            optimizer_G.zero_grad()
            loss_obj.accumulate_gradients(
                phase='Gmain',
                real_img=img,
                mask=mask,
                real_c=gen_c,
                gen_z=gen_z,
                gen_c=gen_c,
                sync=False,
                gain=1.0,
                seg_map=seg
            )
            optimizer_G.step()
        
        # 计算损失和占比（评估模式）
        with torch.no_grad():
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
            vutils.save_image(img, f'MAT/test_results/input_{i}.png', normalize=True, value_range=(-1, 1))
            vutils.save_image(mask, f'MAT/test_results/mask_{i}.png', normalize=True, value_range=(0, 1))
if __name__ == "__main__":
    test_loss_balance()
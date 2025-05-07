import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
import logging
import os
from torch.utils.data import DataLoader
from datasets.dataset_512 import ImageFolderMaskDataset as Dataset512

# 设置日志
logging.basicConfig(filename='test_loss_balance.log', level=logging.DEBUG, format='%(message)s')

# VGG 模型
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].eval().cuda()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, fake, real, mask):
        fake = self.normalize(fake)
        real = self.normalize(real)
        fake_features = self.vgg(fake)
        real_features = self.vgg(real)
        vgg_loss = F.mse_loss(fake_features, real_features)
        
        # 前景和背景感知损失
        fg_mask = mask
        bg_mask = 1 - mask
        fg_fake = fake * fg_mask
        fg_real = real * fg_mask
        bg_fake = fake * bg_mask
        bg_real = real * bg_mask
        fg_loss = F.mse_loss(fg_fake, fg_real) if fg_mask.sum() > 0 else torch.tensor(0.0).cuda()
        bg_loss = F.mse_loss(bg_fake, bg_real) if bg_mask.sum() > 0 else torch.tensor(0.0).cuda()
        
        # 语义加权损失（假设基于分割图）
        sem_weighted_loss = F.mse_loss(fake_features * mask, real_features * mask)
        
        total_pcp_loss = vgg_loss + fg_loss + bg_loss + sem_weighted_loss
        return fg_loss, bg_loss, vgg_loss, sem_weighted_loss, total_pcp_loss

# 权重调整函数
def adjust_weights(adv_loss, pcp_loss, sem_loss, adv_weight, pcp_weight, sem_weight, target_ratios=(0.5, 0.25, 0.25)):
    total = adv_loss * adv_weight + pcp_loss * pcp_weight + sem_loss * sem_weight
    if total == 0:
        return adv_weight, pcp_weight, sem_weight
    
    actual_ratios = [
        (adv_loss * adv_weight) / total,
        (pcp_loss * pcp_weight) / total,
        (sem_loss * sem_weight) / total
    ]
    logging.debug(f"实际占比: adv={actual_ratios[0]*100:.2f}%, pcp={actual_ratios[1]*100:.2f}%, sem={actual_ratios[2]*100:.2f}%")
    
    # 调整权重
    for i, (actual, target) in enumerate(zip(actual_ratios, target_ratios)):
        if actual < target * 0.5:
            if i == 0:
                adv_weight *= 1.1
            elif i == 1:
                pcp_weight *= 1.5
            else:
                sem_weight *= 2.0
        elif actual > target * 1.5:
            if i == 0:
                adv_weight *= 0.9
            elif i == 1:
                pcp_weight *= 0.7
            else:
                sem_weight *= 0.5
    
    # 归一化权重
    total_weight = adv_weight + pcp_weight + sem_weight
    adv_weight /= total_weight
    pcp_weight /= total_weight
    sem_weight /= total_weight
    
    logging.debug(f"调整后权重: adv={adv_weight:.2f}, pcp={pcp_weight:.2f}, sem={sem_weight:.2f}")
    return adv_weight, pcp_weight, sem_weight

# 主测试函数
def test_loss_balance(generator, discriminator, dataset, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 损失函数
    adv_criterion = nn.BCEWithLogitsLoss()
    sem_criterion = nn.CrossEntropyLoss(ignore_index=0)
    pcp_criterion = VGGPerceptualLoss()
    
    # 初始权重
    adv_weight, pcp_weight, sem_weight = 0.1, 10.0, 50.0
    
    generator.eval()
    discriminator.eval()
    
    for idx, (real_img, mask, seg) in enumerate(dataloader):
        real_img = real_img.cuda().float()  # 确保 float32
        mask = mask.cuda().float()
        seg = seg.cuda().long()
        
        logging.debug(f"样本 {idx} - 原始图像尺寸: {real_img.shape[-2]}x{real_img.shape[-1]}")
        logging.debug(f"掩码 原始值: {torch.unique(mask.cpu())}")
        logging.debug(f"分割图 类别: {torch.unique(seg.cpu())}")
        
        # 生成图像
        fake_img = generator(real_img, mask)
        
        # 对抗损失
        real_pred = discriminator(real_img)
        fake_pred = discriminator(fake_img)
        adv_loss = adv_criterion(fake_pred, torch.ones_like(fake_pred))
        
        # 感知损失
        fg_loss, bg_loss, vgg_loss, sem_weighted_loss, total_pcp_loss = pcp_criterion(fake_img, real_img, mask)
        logging.debug(f"前景感知损失: {fg_loss:.4f}, 背景感知损失: {bg_loss:.4f}")
        logging.debug(f"VGG特征损失: {vgg_loss:.4f}, 语义加权损失: {sem_weighted_loss:.4f}, 总感知损失: {total_pcp_loss:.4f}")
        
        # 语义损失
        sem_output = generator.semantic_head(fake_img)  # 假设生成器有语义头
        sem_loss = sem_criterion(sem_output, seg)
        logging.debug(f"语义损失（归一化后）: {sem_loss:.4f}")
        
        # 加权感知损失
        weighted_pcp_loss = total_pcp_loss
        logging.debug(f"加权感知损失: {weighted_pcp_loss:.4f}")
        
        # 总损失
        total_loss = adv_weight * adv_loss + pcp_weight * weighted_pcp_loss + sem_weight * sem_loss
        
        # 权重调整
        adv_weight, pcp_weight, sem_weight = adjust_weights(
            adv_loss, weighted_pcp_loss, sem_loss, adv_weight, pcp_weight, sem_weight
        )
        
        # 日志
        logging.debug(f"对抗损失: {adv_loss:.4f}, 感知损失: {weighted_pcp_loss:.4f}, 语义损失: {sem_loss:.4f}")
        logging.debug(f"总损失: {total_loss:.4f}")
        
        # 保存生成图像
        fake_img = (fake_img + 1) / 2.0  # 假设生成器输出 [-1, 1]
        vutils.save_image(fake_img, f"{output_dir}/sample_{idx}.png")
        
        # 测试日志
        if idx % 10 == 0:
            logging.debug(f"[TEST] 对抗损失: {adv_loss:.4f}, 占比: {(adv_loss*adv_weight/total_loss)*100:.2f}% (目标: 50%)")
            logging.debug(f"[TEST] 感知损失: {weighted_pcp_loss:.4f}, 占比: {(weighted_pcp_loss*pcp_weight/total_loss)*100:.2f}% (目标: 25%)")
            logging.debug(f"[TEST] 语义损失: {sem_loss:.4f}, 占比: {(sem_loss*sem_weight/total_loss)*100:.2f}% (目标: 25%)")
            logging.debug(f"[TEST] 总损失: {total_loss:.4f}")
            logging.debug("-" * 50)
        
        # 仅测试少量样本
        if idx >= 10:
            break

if __name__ == "__main__":
    # 假设你的生成器和判别器已定义
    generator = YourGenerator().cuda()
    discriminator = YourDiscriminator().cuda()
    
    # 数据集
    dataset = Dataset512(image_dir="F:/MAT_project/MAT/data/train_images",
                         mask_dir="F:/MAT_project/MAT/data/train_images/masks",
                         seg_dir="F:/MAT_project/MAT/data/segmentations/train")
    
    # 运行测试
    test_loss_balance(generator, discriminator, dataset)
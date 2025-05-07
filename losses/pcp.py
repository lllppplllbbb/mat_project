import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from losses.vggNet import VGGFeatureExtractor

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights={'conv1_2': 1/16, 'conv2_2': 1/8, 'conv3_3': 1/4, 'conv4_3': 1/2, 'conv5_4': 1}, norm_img=True, criterion='mse'):
        super().__init__()
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys())).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.layer_weights = layer_weights
        self.norm_img = norm_img
        self.criterion_type = criterion
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif criterion == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif criterion == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported")

    def forward(self, x, gt, seg=None):
        if self.norm_img:
            x = (x + 1.) * 0.5  # 归一化到[0,1]
            gt = (gt + 1.) * 0.5

        # 验证分割图
        if seg is not None:
            print(f"[DEBUG] 分割图形状: {seg.shape}, 唯一值: {torch.unique(seg).cpu().numpy()}")
            if torch.all(seg == 0):
                print("[WARNING] 分割图全为背景，无前景像素！")

        # 提取VGG特征
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        vgg_loss = 0
        for k in x_features.keys():
            feat_pixels = x_features[k].shape[2] * x_features[k].shape[3]
            if self.criterion_type == 'fro':
                layer_loss = torch.norm(x_features[k] - gt_features[k], p='fro', dim=(1, 2, 3)) / (feat_pixels * 100)  # 增强归一化
            else:
                layer_loss = self.criterion(x_features[k], gt_features[k]).mean(dim=(1, 2, 3)) / (feat_pixels * 100)  # 增强归一化
            vgg_loss += (layer_loss * self.layer_weights[k]).mean()

        # 语义加权损失
        semantic_loss = 0
        fg_loss = torch.tensor(0.0, device=x.device)
        bg_loss = torch.tensor(0.0, device=x.device)
        if seg is not None:
            # seg: [B, 1, H, W]，值为类别ID，背景为0，前景为非0
            foreground_mask = (seg > 0).float()
            background_mask = (seg == 0).float()
            # 计算前景和背景像素数量
            fg_pixels = foreground_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            bg_pixels = background_mask.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            # 计算像素级 MSE 损失
            pixel_loss = self.criterion(x, gt) / 100  # 降低像素损失量级
            fg_loss = (pixel_loss * foreground_mask).sum(dim=(2, 3)) / fg_pixels.squeeze()
            bg_loss = (pixel_loss * background_mask).sum(dim=(2, 3)) / bg_pixels.squeeze()
            # 前景权重2.0，背景1.0
            semantic_loss = 2.0 * fg_loss.mean() + 1.0 * bg_loss.mean()
            print(f"[DEBUG] 前景像素数: {fg_pixels.sum().item()}, 背景像素数: {bg_pixels.sum().item()}")
            print(f"[DEBUG] 前景感知损失: {fg_loss.mean().item():.4f}, 背景感知损失: {bg_loss.mean().item():.4f}")

        # 结合 VGG 和语义损失
        total_loss = vgg_loss + semantic_loss
        print(f"[DEBUG] VGG特征损失: {vgg_loss.item():.4f}, 语义加权损失: {semantic_loss.item():.4f}, 总感知损失: {total_loss.item():.4f}")
        return total_loss, x_features
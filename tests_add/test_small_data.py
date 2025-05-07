import torch
import numpy as np
from datasets.dataset_512 import ImageFolderMaskDataset
from losses.loss import TwoStageLoss, semantic_loss
from PIL import Image

D = ImageFolderMaskDataset(path='F:/MAT_project/MAT/data/train_images', 
                          seg_dir='F:/MAT_project/MAT/data/segmentations/train', 
                          mask_dir='F:/MAT_project/MAT/data/train_images/masks', 
                          resolution=512)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x

# 临时用随机图像，建议替换为真实生成器
class Generator(torch.nn.Module):
    def __init__(self, z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.conv = torch.nn.Conv2d(img_channels+1, img_channels, 3, padding=1)
    def forward(self, images_in, masks_in, z, c, **kwargs):
        x = torch.cat([images_in, masks_in], dim=1)
        return self.conv(x)

G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(device).eval()
loss_obj = TwoStageLoss(device=device, G_mapping=DummyModel().to(device), 
                       G_synthesis=DummyModel().to(device), D=DummyModel().to(device), 
                       pcp_ratio=0.25, sem_ratio=0.25)

for i in range(min(50, len(D))):
    img, mask, seg, _ = D[i]
    img = torch.from_numpy(img).to(device).unsqueeze(0) / 127.5 - 1.0
    mask = torch.from_numpy(mask).to(device).unsqueeze(0).float()
    seg = torch.from_numpy(seg).to(device).unsqueeze(0).float()
    z = torch.randn(1, G.z_dim, device=device)
    with torch.no_grad():
        pred = G(img, mask, z, torch.zeros([1, G.c_dim], device=device))
    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, img, seg)
    sem_loss = semantic_loss(pred, img, seg)
    mask_ratio = mask.mean().item()
    print(f"样本 {i} - 感知损失: {pcp_loss.item():.4f}, 语义损失: {sem_loss.item():.4f}, 掩码面积比例: {mask_ratio:.4f}")
    print(f"样本 {i} - 类别: {np.unique(seg.cpu().numpy())}")
    pred_img = (pred.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    Image.fromarray(pred_img[0].cpu().numpy(), 'RGB').save(f"pred_{i}.png")
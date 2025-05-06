import torch
import numpy as np
from datasets.dataset_512_val import ImageFolderMaskDataset
from losses.loss import TwoStageLoss
from losses.loss import TwoStageLoss, semantic_loss

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = ImageFolderMaskDataset(path='F:/MAT_project/MAT/data/train_images', seg_dir='F:/MAT_project/MAT/data/segmentations/train', resolution=512)
loss_obj = TwoStageLoss(
    device=device,
    G_mapping=DummyModel().to(device),
    G_synthesis=DummyModel().to(device),
    D=DummyModel().to(device),
    pcp_ratio=0.6,
    sem_ratio=0.4
)

for i in range(min(10, len(D))):
    img, mask, seg, _ = D[i]
    # 将NumPy数组转换为PyTorch张量
    img = torch.from_numpy(img).to(device).unsqueeze(0)
    seg = torch.from_numpy(seg).to(device).unsqueeze(0)
    mask = torch.from_numpy(mask).to(device).unsqueeze(0)
    pred = torch.randn(1, 3, 512, 512, device=device)
    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, img, seg)
    sem_loss = semantic_loss(pred, img, seg)
    print(f"样本 {i} - 感知损失: {pcp_loss.item():.4f}, 语义损失: {sem_loss.item():.4f}")
    print(f"样本 {i} - 类别: {np.unique(seg.cpu().numpy())}, 掩码值: {np.unique(mask.cpu().numpy())}")
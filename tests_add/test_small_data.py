import torch
from datasets.dataset_512 import ImageFolderMaskDataset
from losses.loss import TwoStageLoss

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = ImageFolderMaskDataset(path='F:/MAT_project/MAT/data/train_images', seg_dir='F:/MAT_project/MAT/data/segmentations/train')
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
    img = img.unsqueeze(0).to(device)
    seg = seg.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    pred = torch.randn(1, 3, 512, 512, device=device)
    pcp_loss = loss_obj.perceptual_loss_with_weights(pred, img, seg)
    sem_loss = loss_obj.semantic_loss(pred, img, seg)
    print(f"样本 {i} - 感知损失: {pcp_loss.item():.4f}, 语义损失: {sem_loss.item():.4f}")
    print(f"样本 {i} - 类别: {np.unique(seg.cpu().numpy())}, 掩码值: {np.unique(mask.cpu().numpy())}")
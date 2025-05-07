import torch
import torchvision.utils as vutils
from datasets.dataset_512 import ImageFolderMaskDataset
import matplotlib.pyplot as plt
import os

def generate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageFolderMaskDataset(
        path='F:/MAT_project/MAT/data/train_images',
        resolution=512,
        max_size=10,
        seg_dir='F:/MAT_project/MAT/data/segmentations/train'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 假设加载预训练模型
    G = torch.nn.Module().to(device)  # 替换为实际模型
    G.eval()
    
    os.makedirs('MAT/generated', exist_ok=True)
    for i, (img, mask, seg, _) in enumerate(dataloader):
        img, mask = img.to(device), mask.to(device)
        with torch.no_grad():
            pred = G(img * (1 - mask), mask)
        # 可视化
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('输入图像')
        plt.subplot(1, 3, 2)
        plt.imshow(mask.squeeze(0).squeeze(0).cpu().numpy(), cmap='gray')
        plt.title('掩码')
        plt.subplot(1, 3, 3)
        plt.imshow(pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        plt.title('补全结果')
        plt.savefig(f'MAT/generated/output_{i}.png')
        plt.close()
        vutils.save_image(pred, f'MAT/generated/pred_{i}.png', normalize=True)
        if i >= 4: break

if __name__ == "__main__":
    generate()
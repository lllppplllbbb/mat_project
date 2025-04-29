# MAT训练报告（2025年4月22日）

- **模型**：Mask-Aware Transformer (MAT)
- **分辨率**：128×128
- **批次大小**：1
- **总训练量**：1000 kimg (tick1, kimg4完成)
- **优化器**：Adam (lr=0.002, betas=(0, 0.99), eps=1e-8)
- **损失函数**：TwoStageLoss (r1_gamma=10)
- **数据集**：
  - 训练：10张 (`/kaggle/input/pascal-voc-mat/train_images`)
  - 验证：5张 (`/kaggle/input/pascal-voc-mat/test_images`)
  - 掩码：128×128，0/255
- **数据增强**：xflip, rotate90, scale, brightness, contrast, saturation
- **生成器**：
  - 类：`networks.mat.Generator`
  - z_dim=512, w_dim=512
  - 映射层：8
  - 通道：base=32768, max=512
- **判别器**：
  - 类：`networks.mat.Discriminator`
  - 通道：base=32768, max=512
  - mbstd_group_size=8
- **环境**：
  - Kaggle T4 x2 (单卡运行)
  - PyTorch 2.5.1, Python 3.11
  - CUDA 10.2/11.x
- **训练命令**：

  ```bash
  python train.py --outdir /kaggle/working/output --data /kaggle/input/pascal-voc-mat/train_images --data_val /kaggle/input/pascal-voc-mat/test_images --gpus 1 --batch 1 --kimg 1000 --cfg places256
  ```
- **输出**：
  - 路径：`/kaggle/working/output/00006-train_images-places256-kimg1000-batch1`
  - 检查点：`network-snapshot-000000.pkl`
  - 生成图像：`/output/generated/2008_000002.jpg` 等5张
- **评估指标**（5张验证图）：
  - PSNR: 9.57
  - SSIM: 0.54
  - L1: 0.23
- **备注**：
  - 训练tick1 (kimg4)完成，1000 kimg未全部跑完。
  - 修复`generate_image.py`，适配Kaggle路径。
  - DeepLabv3生成语义分割图。
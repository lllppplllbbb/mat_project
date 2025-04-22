# 训练参数记录（2025-04-22）
## 模型信息
- **模型**：Mask-Aware Transformer（MAT，基于 LoRA 增强的 GAN）
- **分辨率**：128×128
- **批次大小**：1
- **优化器**：Adam (学习率=0.002, betas=(0, 0.99))
- **损失函数**：TwoStageLoss
- **训练量**：
  - 初始：1000 kimg
  - 补充：50 kimg
- **随机种子**：240
## 数据集
- **训练集**：`/kaggle/input/pascal-voc-mat/train_images`
- **验证集**：`/kaggle/input/pascal-voc-mat/test_images`
- **掩码路径**：
  - 训练：`/kaggle/input/pascal-voc-mat/train_images/masks`
  - 验证：`/kaggle/input/pascal-voc-mat/test_images/masks`
- **数据增强**：
  - 亮度、对比度、饱和度调整
  - 旋转、缩放
- **掩码生成**：RandomMask（hole_range=\[0, 1\]）
## 训练环境
- **平台**：Kaggle
- **GPU**：T4（单卡）
- **PyTorch 版本**：2.5.1+cu124
- **配置文件**：
  - 路径钩子：Kaggle 路径配置（训练/验证数据、掩码、调试模式）
  - 数据集：`dataset_512.py`（插入 RandomMask 导入，索引循环）
  - 评估：`frechet_inception_distance.py`（num_gen=10）
## 生成参数
- **生成脚本**：`generate_image.py`
- **网络权重**：`/kaggle/working/output/00006-train_images-places256-kimg1000-batch1/network-snapshot-000000.pkl`
- **生成分辨率**：128×128
- **截断系数**：`truncation_psi=1.0`
- **噪声模式**：`const`
- **输出路径**：`/kaggle/working/output/generated/*.jpg`
## 评估结果
- **评估脚本**：`cal_psnr_ssim_l1.py`
- **验证集**：5 张图像（2008_000002.jpg, 2008_000003.jpg, 2008_000007.jpg, 2008_000009.jpg, 2008_000016.jpg）
- **指标**：
  - PSNR: 9.57
  - SSIM: 0.54
  - L1: 0.23
- **输出文件**：`/kaggle/working/output/psnr_ssim_l1_results.txt`
## 语义分割
- **模型**：DeepLabv3 (ResNet101 预训练)
- **输出**：`/kaggle/working/output/*_seg.png`
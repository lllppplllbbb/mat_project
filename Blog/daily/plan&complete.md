## 4.18
- 确立了新的基础模型 MAT
- 明确了三个创新点（两个主要模块、一个损失函数）
- 本地克隆MAT模型并安装完环境
- 明日计划 1.学习mat模型 2.学习loss函数 
## 4月19日学习进展
- 学习了MAT模型架构，阅读论文第3节和`mat.py`，整理了`framework.md`（包括Encoder、BasicLayer、Decoder、FirstStage、掩码机制、U型结构）。
- 学习了损失函数，阅读论文第3.3节和`loss.py`，更新了`Loss.md`（包括L1、感知、对抗、R1正则化、总损失）。
- 画了MAT流程图（Mermaid格式），梳理了从输入到输出的代码逻辑。
- 明日计划：
  1. 跑小规模训练（PASCAL VOC 10张），观察损失和补全效果。
  2. 继续读`train.py`和`data/dataset.py`，理解训练和数据加载。
  3. 准备DeepLabv3，为语义加权损失和自适应掩码做准备。
## 4.20
- 下载了PASCAL VOC 2012数据集，准备开始训练。
- 改mask_generator_512_small.py，生成10+5张掩码，test_masks.py验证（128x128，0/255）。
## 4.21
- 验证10+5张128x128图像和掩码（`/kaggle/input/pascal-voc-mat`）。
- 调整Kaggle文件结构（`datasets/`, `losses/`, `networks/`）。
- 解决`dnnlib` 404，用`dnnlib.py`（`EasyDict`）。
- 尝试PyTorch 1.8.1+cu102、1.10.0+cu102、1.7.0+cu101，均失败（pip源无旧版）。
- 发现`train.py`可能缺依赖（如`training_loop.py`）。
- **问题**：
  - PyTorch版本不可用。
  - 训练依赖不完整。
- **明日计划**：
  1. 解决PyTorch（1.13.0+cu117或清华源）。
  2. 补齐`train.py`依赖。
  3. 跑训练（128x128，kimg=1）。
  4. 加打印（`training_loop.py`, `loss.py`）。
  5. 跑DeepLabv3。
## 4.22
1. **训练**：
   - 成功运行 `train.py`，完成 1000 kimg 训练（T4 单卡，tick 1 kimg 4）。
   - 配置 Kaggle 路径钩子，适配数据集路径。
2. **生成**：
   - 编写并修复 `generate_image.py`，实现自动图像修复。
   - 输出 5 张修复图像到 `/kaggle/working/output/generated`。
3. **语义分割**：
   - 使用 DeepLabv3 生成 5 张分割图像（`2008_000002_seg.png` 等）。
4. **评估**：
   - 修复 `cal_psnr_ssim_l1.py`，成功输出 PSNR: 9.57, SSIM: 0.54, L1: 0.23。
5. **配置修改**：
   - 添加路径钩子，修补数据集、损失和训练循环模块。
   - 修改 `dataset_512.py`（插入 RandomMask 导入，添加索引循环）。
   - 调整 `frechet_inception_distance.py`（num_gen=10）。
6. **学习**：
   - 学会查看输出路径、保存模型、批量重命名。
- 问题
   - **训练效率**：T4 单卡训练较慢（1000 kimg 耗时长），未充分利用双卡。
   - **生成质量**：PSNR 9.57 和 SSIM 0.54 较低，修复图像质量需优化。
   - **评估细节**：仅输出平均指标，需检查每张图像的具体表现。
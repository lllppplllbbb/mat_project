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
- 明日计划
- **目标**：优化训练，跑256x256，改进指标。
- **任务**（4-6小时）：
  1. 保存Notebook，下载`output.zip`（0.5小时）。
  2. 试T4双卡或P100，跑1000 kimg（2小时）。
  3. 调整`kimg=10`，跑256x256（1小时）。
  4. 优化模型（加DeepLabv3语义引导，0.5小时）。
  5. 评估PSNR/SSIM/L1，更新`notes.md`（0.5小时）。
- **注意**：
  - 检查T4双卡配置（`--gpus 2`）。
  - 若P100更快，切换加速器。
  - 保存检查点和生成图像。
  ## 4.23
  - 数据集准备完成（200训练+50验证，掩码和分割图齐全）。
  - 代码改动完成（`dataset_512.py`, `dataset_512_val.py`, `train.py`, `loss.py`, `metric_main.py`, `training_loop.py`, `generate_image.py`）。
  - 测试数据加载成功，形状正确（512x512）。
  - 未运行训练，计划4月24日完成。
  ## 4月28日



## 完成的工作
1. **数据集与掩码生成**
   - **copy_voc.py**：成功复制200张训练图和50张测试图，处理了分割图中的255值，确保类别分布合理（0-20全覆盖）。
   - **mask_generator_512_small.py**：优化掩码生成，移除无意义的220处理，确认输出值为0/255（二值化）。训练和测试掩码平均值分别为0.78和0.77，适合文物修复。
   - **test_dataset.py**：增强掩码检查，确认数据集大小400（含翻转增强），类别分布合理，但掩码值为[0. 1.]（浮点型，需修复为0/1）。
   - **改动**：更新`mask_generator_512_small.py`和`dataset_512.py`，确保掩码二值性。

2. **语义加权感知损失**
   - **问题**：运行`test_semantic_loss.py`报`TypeError: forward() missing 1 required positional argument: 'gt'`，因`loss.py`中`self.pcp(target)`调用错误。
   - **修复**：更新`loss.py`，使用`gt_features`避免重复调用`self.pcp`。同步更新`pcp.py`和`vggNet.py`的层命名（`relu4_2`, `relu5_4`）。
   - **待验证**：运行`test_semantic_loss.py`，检查前景/背景损失差异。

3. **总损失平衡**
   - **问题**：`test_loss_balance.py`因感知损失错误失败。
   - **修复**：依赖`loss.py`修复，目标损失比例为对抗50%、感知25%、语义25%。
   - **待验证**：运行`test_loss_balance.py`，检查比例。

4. **FID评估**
   - **问题**：运行`fid50_full,fid1k_full`无输出，`fid_debug.log`为空，可能因数据加载失败或日志配置错误。
   - **修复**：更新`metric_main.py`和`metric_utils.py`，添加数据加载和Inception输入日志，强制`cache=False`。
   - **待验证**：重新运行FID命令，检查`fid_debug.log`和终端输出。

5. **小规模测试**
   - **状态**：`test_small_data.py`已添加调试，但因感知损失问题未运行。
   - **计划**：修复感知损失后，测试10张图像的损失和类别。

6. **全数据集准备**
   - **计划**：更新`copy_voc.py`支持PASCAL VOC全集（1,464训练+1,449验证），待前置问题解决。
   - **掩码生成**：确认命令：
     ```bash
     & f:/MAT_project/mat_env/Scripts/python.exe f:/MAT_project/datasets/mask_generator_512_small.py --img_dir "F:/MAT_project/MAT/data/train_images" --mask_dir "F:/MAT_project/MAT/data/masks/train"
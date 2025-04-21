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
## 4.20
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
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
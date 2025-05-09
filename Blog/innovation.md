# 创新点描述与实现思路

## 创新点1：语义引导的自适应掩码生成机制

### 描述
这个是让模型别瞎补图，先看懂图像里啥重要。比如修文物壁画，模型得知道人物和图案比背景重要。我们用DeepLabv3给图像分块，标出哪儿是关键区域，然后根据这些信息动态生成掩码，告诉模型优先修这些地方，补出来的图更像回事。

### 怎么实现
先拿个现成的DeepLabv3模型，丢进图像，它会吐出一张图，标明哪儿是文物图案、哪儿是背景。然后我们写个小程序，根据这些标记决定掩码怎么画，比如让重要区域的掩码多遮点，背景少遮点。接着把这掩码喂给MAT模型，让它按这个掩码补图。整个过程就像给模型发个"重点补这儿"的指令，MAT收到后就知道咋干了。

## 创新点2：多模态语义引导的特征融合模块

### 描述
这个是给模型喂更多信息，让它修图更聪明。光看图像不够，我们把边缘线条、纹理细节、语义分割这些信息都塞给模型。比如修风景图，边缘帮模型知道哪儿是树，语义分割告诉它哪儿是天空，纹理让草地更真实，混在一起效果更自然。

### 怎么实现
我们得建几个小模块，一个抓图像的边缘（用现成的边缘检测工具），一个抓纹理（用预训练模型提特征），一个抓语义分割（还是用DeepLabv3）。然后弄个"搅拌机"，把这些信息混起来，方法是用个注意力机制，让模型自己挑重要的信息用。把这堆混好的信息喂给MAT的Transformer部分，它修图时就能看全大局，补得更细致。

## 创新点3：语义加权的混合损失函数

### 描述
这个是改模型的评分标准，让它更注重重要区域。原来模型用像素差评分，容易补得模糊。我们加了感知损失和对抗损失，让补的图更真实，还根据语义分割给重要区域（比如文物图案）更高的权重，模型就更用心修关键部分。

### 怎么实现
先用DeepLabv3标出图像的重点区域，比如文物图案标个高分，背景标低分。然后改MAT的评分规则，原先只看像素差（MSE），我们再加个感知损失（用VGG模型比特征），再加个对抗损失（让补的图更像真的）。重点是给这些损失加权重，重要区域的错误罚重一点，背景轻一点。改完后，MAT训练时就知道得先把重点区域修好。

具体来说：
1. **语义分割权重生成**：
   - 用DeepLabv3处理输入图像，得到每个像素的语义标签
   - 根据语义标签的重要性，给每个区域分配权重值（比如人物区域权重1.5，背景区域权重0.8）
   - 生成一个与图像大小相同的权重图

2. **混合损失函数构建**：
   - 像素级损失：计算原始图像和生成图像的MSE，但乘以权重图
   - 感知损失：用预训练的VGG网络提取原始图像和生成图像的特征，计算特征差异，同样乘以权重图
   - 对抗损失：训练一个判别器网络，让它区分真实图像和生成图像，提高生成图像的真实感

3. **损失函数整合**：
   - 总损失 = α×加权像素损失 + β×加权感知损失 + γ×对抗损失
   - 其中α、β、γ是可调节的超参数，控制各部分损失的相对重要性

4. **训练过程优化**：
   - 在每个训练批次中，先计算语义分割，生成权重图
   - 前向传播生成修复图像
   - 计算加权混合损失
   - 反向传播更新模型参数，让模型更关注重要区域
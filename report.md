弱增强和强增强

论文描述为50%的概率水平翻转，12.5%的概率移动（即移动4个像素）

强增强：简化版，实现rand

MixMatch 是 FixMatch 的"前辈"，来自 Berthelot 等人 2019 年的论文《MixMatch: A Holistic Approach to Semi-Supervised Learning》（NeurIPS 2019），也是 Google 这条线的工作。FixMatch 论文里把它作为重要的对比 baseline。

**1. 给无标签数据猜一个"软标签"（guessed label）**
对每张无标签图片做 K 次（论文里 K=2）简单数据增强（比如随机裁剪、翻转），把这 K 个增强版本都喂进模型，把预测的概率分布**取平均**，得到一个初步的猜测分布。

**2. 锐化（Sharpening）**
把上一步平均出来的概率分布用温度参数 T 做锐化（降低 T 会让分布更"尖"、更接近 one-hot），让猜测标签更有把握、更接近某一类，而不是模糊的软分布。这一步跟 FixMatch 用"置信度阈值直接取硬标签"是不同的思路——MixMatch 是"软化处理 + 调温度"，FixMatch 是"达标就当真，没达标就丢弃"。

**3. MixUp 混合**
把有标签数据（带真实 one-hot 标签）和刚才处理好的无标签数据（带猜测的软标签）混在一起，整体打乱，然后两两用 MixUp 做凸组合：

```
x' = λ·x1 + (1-λ)·x2
y' = λ·y1 + (1-λ)·y2
```

λ 从 Beta 分布里采样。这样训练数据和标签都被揉在一起了。

**最终 loss**：

- 有标签部分：普通交叉熵
- 无标签部分：用**均方误差**（而不是交叉熵）衡量预测和猜测软标签的差距，这样对预测错得离谱的情况惩罚没那么剧烈
- 两部分加权相加，无标签项的权重 λu 还要配合一个**线性 ramp-up（逐渐增大）**的调度策略，训练初期权重很小，避免一开始模型还很差时被错误的猜测标签带偏

**和 FixMatch 的关系/区别**，这也是 FixMatch 论文的核心卖点：

- MixMatch 需要调的超参数很多（K、锐化温度 T、MixUp 的 α、ramp-up 调度…），训练流程也更复杂
- FixMatch 把这套简化成"弱增强出伪标签 + 置信度阈值过滤 + 强增强算 loss"，去掉了 MixUp、锐化、软标签这些环节，超参数少了很多，反而在标签极少的场景（比如 CIFAR-10 只有 40 张标签）效果明显更好、更稳定
- 论文里报的数字大致是：CIFAR-10 用 250 张标签时，MixMatch 错误率在 11% 左右，FixMatch 能压到 5% 左右；标签数越少，FixMatch 的优势越明显

EMA model
使用了ema，有teacher模型正常梯度下降，学生模型指数移动平均

$v_t=\beta*v_{t-1}+(1-\beta)*\theta_t$

他表示了过去 $1/(1-\beta)$ 的平均


发现后期出现了在unmask count波动巨大的现象

```
Step [4800/8192], Loss: 0.1612, Loss_x: 0.0040, Loss_u: 0.1572, Test Accuracy: 43.92%, Time: 1.19s
Unmask counts : [5455 1491  902 2564 4085  430 3545 2264  942 2723]
Step [4900/8192], Loss: 0.1166, Loss_x: 0.0010, Loss_u: 0.1156, Test Accuracy: 43.22%, Time: 1.19s
Unmask counts : [6225 1771  954 3311 4437    5 3820 2100  761 2582]
Step [5000/8192], Loss: 0.1219, Loss_x: 0.0011, Loss_u: 0.1208, Test Accuracy: 44.09%, Time: 1.20s
Unmask counts : [5685 1524 1447 3462 4321    1 3489 2237  914 2582]
Step [5100/8192], Loss: 0.1291, Loss_x: 0.0011, Loss_u: 0.1280, Test Accuracy: 44.38%, Time: 1.19s
Unmask counts : [5520 1862 1406 3172 4324    6 3542 2365  789 2459]
Step [5200/8192], Loss: 0.1125, Loss_x: 0.0010, Loss_u: 0.1115, Test Accuracy: 42.33%, Time: 1.19s
Unmask counts : [5605 1528 1466 3498 4233    2 3911 2227  755 2843]
Step [5300/8192], Loss: 0.1236, Loss_x: 0.0013, Loss_u: 0.1223, Test Accuracy: 42.40%, Time: 1.19s
Unmask counts : [6153 1652 1469 1853 3835  592 4518 2351  738 2705]
```

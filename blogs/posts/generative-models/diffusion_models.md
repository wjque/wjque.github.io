# Diffusion Models

在学习扩散模型的原理之前，我们需要了解扩散模型是如何发展而来的。本文的讨论框架参考 [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) 这篇论文提出的观点，此外还会结合语言模型的发展进行讨论。

## 自编码器 (AutoEncoder)

在自然语言处理领域，如何将文本进行编码是一项非常重要的任务，现在的大语言模型在编码阶段需要做以下几件事情：

- 分词 (**tokenize**)：经过 `tokenizer` 将一句话转化为一个单词序列。
- 映射 (**word to tokenID**)：查询词汇表 `vocab`，将单词序列转化为 `tokenID` 序列。
- 嵌入 (**tokenID to embedding**)：模型的输入端有一个嵌入矩阵，形状为 $(\text{len}(vocab), \text{dim})$，会将 `tokenID` 序列转化为嵌入向量序列。

其中分词器的设计属于自然语言处理的经典任务，不同 `LLM` 可以使用相同的 `tokenizer` 和 `vocab`，但一般有不同的嵌入矩阵，因为这个矩阵是随模型一同训练的，但是训练结束后，这个嵌入矩阵就是固定的一张 "码表"，能够把一个单词的 `tokenID` 转化为一个隐空间中的向量。

训练的方式有几种：

- 自回归语言模型：基于当前的词预测未来的词 (**GPT**; **LLaMA**; **DeepSeek**)
- 掩码语言模型：基于周围的词预测被掩码的词 (**BERT**)

所以语言模型对文本的编码能力是在训练过程中自动学习到的，视觉方面的任务也可以使用类似的思想，**让模型自动学习编码和解码图像**，这就是自编码器 (`AutoEncoder`) 的由来。

关于为什么要考虑编码图像这个任务，**Calvin Luo** 在论文中引用的柏拉图洞穴之喻非常形象地回答了这个问题：

> 有一群从小就被困在不能转身的洞穴中的人，只能看到墙壁上的影子，他们认为这些影子就是现实，直到有一天他们逃离了洞穴，看到太阳后才明白，影子都是太阳产生的，太阳才是最真实的东西。

我们看到的图像就好比是这些影子，它们也许不是本质，我们要研究图像分布的本质，就必须尝试换一个角度去看待它们。

自编码器的一种典型设计是使用 **U-Net**，通过一个卷积神经网络将图像压缩为更小的特征图，再训练一个对称的卷积神经网络学习将其解码映射回原始图像，从而学到图像的编码方式，训练的方式也很简单，只需要对解码出的图像和原始图像做均方差损失 (**MSE**) 即可。

这种自编码器有以下几种应用：

- 数据降噪与重建：在训练过程中，输入添加一定高斯噪声的图像，经编码和解码后得到的图像仍然与正常的图像做 **MSE**，可以训练一个去噪自编码器模型。
- 特征提取与降维：可以将提取的低维特征用于数据可视化，也可以直接将这些特征作为其他监督学习任务的输入。
- 异常检测：如果自编码器只使用 "正常样本" 训练，那么它只会重建 "正常" 的模式，可以设定一个重建误差阈值，用于判定一张图片是否正常。

当然，这种经典的自编码器也存在一定的问题，因为它偏向于**记忆数据集的模式**，所以一旦使用 **OOD (Out of Domain)** 的样本，重建结果就可能崩溃。因此如果直接用一个和低维特征尺寸相同的向量输入解码器，得到的很可能是乱码而非正常图片。

## 变分自编码器 (Variational AutoEncoder)

为了得到具有生成能力的模型，我们不能将隐空间固定为一个向量，而应该将其**建模为一个分布**（一般把编码得到的向量作为分布的均值，方差固定为一个典型值），这样在生成时就可以从这样一个分布中采样，得到可用于解码出正常图片的隐空间向量，这就是变分自编码器的核心思想。

### 问题阐述 (Problem Formulation)

给定观测数据 $\mathbf{x}$（例如一张 $256\times256$ 的 RGB 图像），我们假设它由一个隐含的真实分布 $p(\mathbf{x})$ 生成。我们的目标是学习一个生成模型 $p_{\boldsymbol{\theta}}(\mathbf{x})$，能够**采样出与真实数据不可区分的新样本**。

直接对 $p(\mathbf{x})$ 在像素空间建模几乎不可能——一张 $256\times256$ 的图像有 $256\times256\times3\approx2\times10^5$ 个维度，高维空间中的分布极其复杂。我们借鉴自编码器降维的思想，引入低维的隐变量 $\mathbf{z} \in \mathbb{R}^d$（$d$ 远小于数据维度），将生成过程分解为两步：

- 从先验分布采样隐变量：$\mathbf{z} \sim p(\mathbf{z})$
- 从隐变量解码生成数据：$\mathbf{x} \sim p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$

于是生成模型的边缘分布为：

$$
p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}
$$

通常设先验为标准高斯 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，而 $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$ 是一个由神经网络 $\boldsymbol{\theta}$ 参数化的条件分布，对于连续图像，通常取高斯分布

$$
\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}), \sigma^2 \mathbf{I})
$$

此时 $\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$ 等价于 **MSE** 损失的负数。

**两个核心困难**：

1. **边缘分布不可计算**：上式中的积分在高维空间中是 $\text{intractable}$ 的——因为我们无法遍历 $\mathbf{z}$ 的所有可能取值。
2. **后验分布不可计算**：给定数据求隐变量的分布 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) = \frac{p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})}{p_{\boldsymbol{\theta}}(\mathbf{x})}$，分母恰是困难的积分。

**解决思路——变分推断 (Variational Inference)**：我们引入一个参数化的分布 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ 作为对真实后验 $p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$ 的近似。在 VAE 中，编码器输出这个近似后验的参数：

$$
q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\big( \mathbf{z}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}), \boldsymbol{\sigma}^2_{\boldsymbol{\phi}}(\mathbf{x}) \cdot \mathbf{I} \big)
$$

其中 $\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x})$ 和 $\boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\mathbf{x})$ 均由编码器网络输出。注意这里用**对角协方差**假设，意味着隐变量的各维度之间相互独立——这既简化了计算，在实验中也被证明足够有效。

用一个比喻来理解：我们手中有大量图像样本（数据），但不知道它们遵循什么生成规律。我们猜测背后有某种 "本质特征"（隐变量），比如人脸图像的本质特征包括肤色、脸型、表情等。编码器试图从一张图推断这些特征的后验分布（均值 = 最可能的值，方差 = 不确定性），解码器则根据这些特征重建图像。训练完成后，我们从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 中随机采样一个 $\mathbf{z}$，喂给解码器，就能得到一张 "凭空创造" 的合理图像。

### Evidence Lower BOund (ELBO)

有了编码器 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ 和解码器 $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$，我们需要一个可优化的目标函数。回顾我们最终想最大化的是对数似然 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$——即模型看待真实数据的 "认可程度"。

推导如下：

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x})
&= \log \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} \\
&= \log \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \cdot \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \, d\mathbf{z} \\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right]
\end{aligned}
$$

由 **Jensen 不等式**（$\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$，因为 $\log$ 是凹函数）：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x}) \geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \right]
$$

将联合分布展开 $p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})$，得到 **Evidence Lower BOund (ELBO)**：

$$
\begin{aligned}
\text{ELBO}
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \big[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) + \log p(\mathbf{z}) - \log q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \big] \\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})} \big[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}) \big] - \text{KL} \big( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\big\|\, p(\mathbf{z}) \big)
\end{aligned}
$$

这个下界的两个组成部分恰好对应了 VAE 设计的两个直觉目标：

1. **重建项** $\mathbb{E}_q [\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})]$：
   - 给定编码得到的 $\mathbf{z}$，解码出的 $\mathbf{x}$ 是否和原图一致？
   - 在 $p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})$ 为高斯分布的假设下，这一项等价于**负 MSE 损失**。
   - 这一项促使编码器提取对重建有用的信息。

2. **KL 正则项** $-\text{KL}(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$：
   - 要求每个样本的近似后验 $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})$ 不要偏离先验 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 太远。
   - 如果每个输入都被编码到完全不同的隐空间区域，从 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 采样的 $\mathbf{z}$ 就可能掉在 "无人区"，解码出乱码。
   - 这一项惩罚隐空间的 "碎片化"，强迫它平滑、集中。

**为什么叫 "Evidence Lower BOund"？** 我们还可以证明：

$$
\log p_{\boldsymbol{\theta}}(\mathbf{x}) - \text{ELBO} = \text{KL}\big( q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \,\big\|\, p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}) \big) \geq 0
$$

即 ELBO 与真实对数似然的差距，恰好是近似后验与真实后验的 KL 散度。**最大化 ELBO 同时在提高 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$（生成质量）和逼近似后验（编码质量）**。

**训练中的重参数化技巧 (Reparameterization Trick)**：

直接从 $\mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\phi}}, \boldsymbol{\sigma}_{\boldsymbol{\phi}}^2)$ 中采样 $\mathbf{z}$ 会阻断梯度反向传播，因为采样操作不可导。解决方案是：

$$
\mathbf{z} = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

随机性被 "剥离" 到 $\boldsymbol{\varepsilon}$ 中，梯度可以正常流经 $\boldsymbol{\mu}_{\boldsymbol{\phi}}$ 和 $\boldsymbol{\sigma}_{\boldsymbol{\phi}}$。这一技巧虽然简单，却是 VAE 能够端到端训练的关键。

**具体例子：手写数字生成 (MNIST)**

用 MNIST 数据集训练 VAE 后，我们可以做以下实验来直观理解隐空间的结构：
- **插值**：取两张数字图像的编码 $\mathbf{z}_A$ 和 $\mathbf{z}_B$，在线段上均匀采样 $\mathbf{z}_t = (1-t)\mathbf{z}_A + t\mathbf{z}_B$ 并解码——你会发现数字从 "3" 平滑地变形为 "8"，中间状态的过渡自然且合理。这说明隐空间是连续的、有语义意义的。
- **属性算术**：研究发现隐空间中存在语义方向，例如 $\mathbf{z}_{\text{戴眼镜}} - \mathbf{z}_{\text{不戴眼镜}}$ 近似代表 "眼镜" 属性，将这个向量加到另一个人的编码上，解码后就能给他 "戴上眼镜"。这说明隐空间不仅连续，而且是**结构化的**。

这些特性是纯 MSE 自编码器不具备的——它只记住了离散的隐向量到图像的映射，隐空间中任意采样几乎必然得到乱码。

## 级联变分自编码器 (Hierarchical)

普通 VAE 只有一个隐变量层 $\mathbf{z}$，但这有什么问题呢？考虑生成一张人脸照片的过程：

1. 首先需要确定整体布局：头的位置、背景颜色、光照方向。
2. 然后确定面部结构：五官轮廓、发型。
3. 最后才填充纹理细节：毛孔、皱纹、发丝。

这些不同粒度 (granularity) 的信息很难被压缩到单一维度的隐向量中同时表达。**级联变分自编码器 (Hierarchical Variational AutoEncoder, HVAE)** 的思想是将隐变量扩展为 $T$ 层，每一层负责一个抽象等级。

### 马尔可夫层级结构

假设我们有 $T$ 个隐变量 $\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_T$，其中 $\mathbf{z}_1$ 最接近数据（细粒度），$\mathbf{z}_T$ 最抽象（粗粒度）。马尔可夫 HVAE 的结构为：

- **编码路径**（从下到上，$\mathbf{x} \to \mathbf{z}_T$）：

  $$
  q_{\boldsymbol{\phi}}(\mathbf{z}_{1:T} \mid \mathbf{x}) = q_{\boldsymbol{\phi}}(\mathbf{z}_1 \mid \mathbf{x}) \prod_{t=2}^{T} q_{\boldsymbol{\phi}}(\mathbf{z}_t \mid \mathbf{z}_{t-1})
  $$

  每一层在前一层的基础上提取更高层次的抽象特征。这和深度卷积网络逐层提取特征的过程十分相似——浅层检测边缘和纹理，深层检测语义概念。

- **解码路径**（从上到下，$\mathbf{z}_T \to \mathbf{x}$）：

  $$
  p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}_{1:T}) = p(\mathbf{z}_T) \prod_{t=2}^{T} p_{\boldsymbol{\theta}}(\mathbf{z}_{t-1} \mid \mathbf{z}_t) \cdot p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}_1)
  $$

  生成过程从最抽象的 $\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，逐层向下生成越来越具体的细节，最终得到图像。

### HVAE 的 ELBO

将单层 VAE 的推导直接推广到 $T$ 层，得到层级 ELBO：

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x})
&\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}_{1:T} \mid \mathbf{x})} \left[ \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}_{1:T})}{q_{\boldsymbol{\phi}}(\mathbf{z}_{1:T} \mid \mathbf{x})} \right] \\
&= \mathbb{E}_{q} \big[ \log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}_1) \big]
   - \text{KL}\big( q_{\boldsymbol{\phi}}(\mathbf{z}_T \mid \mathbf{z}_{T-1}) \,\big\|\, p(\mathbf{z}_T) \big) \\
&\quad - \sum_{t=2}^{T-1} \mathbb{E}_{q} \Big[ \text{KL}\big( q_{\boldsymbol{\phi}}(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \,\big\|\, p_{\boldsymbol{\theta}}(\mathbf{z}_t \mid \mathbf{z}_{t+1}) \big) \Big]
\end{aligned}
$$

这个目标函数的每一层都在做两件事：(1) 重建该层的信息；(2) 保持前后层之间的分布一致。$T$ 层越多，模型对不同抽象等级的表达能力越强，但优化也越困难。

### 从 HVAE 到扩散模型的直觉桥梁

层级 VAE 的思路很好，但实践中训练深层 HVAE 极其困难——$T$ 个隐变量的编码器和解码器都需要学习，KL 散度的积累也容易导致后验坍缩 (posterior collapse)。

**关键洞察**：如果我们放弃学习编码器，而是人工设计一个**固定的编码过程**呢？

具体来说，我们可以把 "编码" 简单地定义为：往图像中逐步添加高斯噪声。

- 第 1 步加一点噪声 → $\mathbf{z}_1$
- 第 2 步再加一点噪声 → $\mathbf{z}_2$
- ...
- 第 $T$ 步加到几乎纯噪声 → $\mathbf{z}_T$

这样编码器不再需要任何参数！我们只需要学习解码器——即学习如何从噪声逐步恢复出干净图像。这正是扩散模型的核心思想。

用一个生活中的例子来理解：把一张照片反复复印（每次复印都引入一些噪点），第 100 次复印件已经面目全非了。但你如果仔细观察每次复印之间的退化规律，理论上可以反推出原图应该是什么样子。扩散模型学做的就是这件事，只不过它的 "复印机" 是我们自己设计的高斯噪声注入过程，其数学性质非常干净。

## Noise based Diffusion Models

### Denoise Diffusion Probability Models (DDPM)

**[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** (DDPM) 是由 **Ho et al.** 在 2020 年提出的，奠定了现代扩散模型的基础框架。它由两个过程组成：一个固定的**前向过程**（破坏数据）和一个可学习的**逆向过程**（恢复数据）。

#### 前向过程 (Forward Process)

前向过程是一个**马尔可夫链**，在 $T$ 步内逐步将数据 $\mathbf{x} = \mathbf{z}_0$ 转化为纯噪声 $\mathbf{z}_T$：

$$
q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) = \mathcal{N}\big( \mathbf{z}_t; \sqrt{1 - \beta_t} \, \mathbf{z}_{t-1}, \beta_t \mathbf{I} \big)
$$

其中 $\beta_t \in (0, 1)$ 是**噪声调度 (noise schedule)**，控制每步加入多少噪声：
- 当 $\beta_t \to 0$：$\mathbf{z}_t \approx \mathbf{z}_{t-1}$（几乎不加噪声）
- 当 $\beta_t \to 1$：$\mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（完全替换为纯噪声）

在前向过程中，我们希望最终 $\mathbf{z}_T$ 几乎完全失去原始数据的信息，即 $q(\mathbf{z}_T \mid \mathbf{x}) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$。

<center class='img'>
<img title="ddpm forward" src="./figs/ddpm/forward_process.png" width="80%">
</center>

**重参数化与一步采样**

高斯分布的性质允许我们**跳过中间步骤**，直接从 $\mathbf{x}$ 计算任意时刻的 $\mathbf{z}_t$。记 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$：

$$
\begin{aligned}
\mathbf{z}_t
&= \sqrt{\alpha_t} \, \mathbf{z}_{t-1} + \sqrt{\beta_t} \, \boldsymbol{\varepsilon}_{t-1} \\
&= \sqrt{\alpha_t \alpha_{t-1}} \, \mathbf{z}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \, \boldsymbol{\varepsilon}_{t-2}' \\
&= \cdots \\
&= \sqrt{\bar{\alpha}_t} \, \mathbf{x} + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\end{aligned}
$$

即：

$$
q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}\big( \mathbf{z}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}, (1 - \bar{\alpha}_t) \mathbf{I} \big)
$$

这是一个非常重要的性质。它意味着训练时**不需要迭代 $t$ 步来生成加噪样本**——给定 $\mathbf{x}$ 和 $t$，一次采样就能得到 $\mathbf{z}_t$。

#### 逆向过程 (Reverse Process)

逆向过程也是一个马尔可夫链，从一个随机噪声 $\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 开始，逐步去噪恢复出图像：

$$
p_{\boldsymbol{\theta}}(\mathbf{z}_{t-1} \mid \mathbf{z}_t) = \mathcal{N}\big( \mathbf{z}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t, t), \sigma_t^2 \mathbf{I} \big)
$$

问题的核心是**如何学习均值 $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)$**。DDPM 的洞见在于：当 $\beta_t$ 足够小时，逆向过程在每一步也可以近似为高斯分布。而我们可以**用前向过程的后验 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$ 作为学习目标**——这个后验是**可解析计算的**（这是扩散模型不同于普通 VAE 的关键优势）。

#### 从 ELBO 推导训练目标

回忆 HVAE 的 ELBO。在 DDPM 中：
- 编码器 $q$ 是固定的（无参数 $\boldsymbol{\phi}$）
- $\mathbf{z}_t$ 的维度与数据 $\mathbf{x}$ 相同（非降维，这也是扩散模型与普通 VAE 的重要区别）
- $p(\mathbf{z}_T) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

将 ELBO 写为 DDPM 的形式：

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x})
&\geq \mathbb{E}_{q} \left[ \log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}_{1:T})}{q(\mathbf{z}_{1:T} \mid \mathbf{x})} \right] \\
&= \mathbb{E}_{q} \Big[ -\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z}_1)
    + \sum_{t=2}^{T} \text{KL}\big( q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \,\big\|\, p_{\boldsymbol{\theta}}(\mathbf{z}_{t-1} \mid \mathbf{z}_t) \big) \\
&\qquad\qquad + \text{KL}\big( q(\mathbf{z}_T \mid \mathbf{x}) \,\big\|\, p(\mathbf{z}_T) \big) \Big]
\end{aligned}
$$

这个分解的推导利用了前向过程的马尔可夫性，将 $\log q(\mathbf{z}_{1:T} \mid \mathbf{x})$ 重新排列为一个可计算的 KL 散度之和。

**核心：可解析的 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$**

对于前向过程的每一步，给定 $\mathbf{z}_t$ 和原始图像 $\mathbf{x}$，$\mathbf{z}_{t-1}$ 的后验也是高斯分布（通过贝叶斯公式和两个高斯分布的乘积性质）：

$$
\begin{aligned}
q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})
&= \mathcal{N}\big( \mathbf{z}_{t-1}; \boldsymbol{\mu}_q(\mathbf{z}_t, \mathbf{x}), \tilde{\beta}_t \mathbf{I} \big) \\
\boldsymbol{\mu}_q(\mathbf{z}_t, \mathbf{x}) &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x} + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{z}_t \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
\end{aligned}
$$

**推导**：利用贝叶斯公式 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \propto q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \, q(\mathbf{z}_{t-1} \mid \mathbf{x})$，将两个高斯密度相乘并配方 (complete the square) 即可得到上述结果。

#### 噪声预测参数化

现在我们让模型 $p_{\boldsymbol{\theta}}(\mathbf{z}_{t-1} \mid \mathbf{z}_t)$ 的方差与 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$ 一致，即 $\sigma_t^2 = \tilde{\beta}_t$。那么每个 KL 散度项简化为：

$$
\text{KL}\big( q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x}) \,\big\|\, p_{\boldsymbol{\theta}}(\mathbf{z}_{t-1} \mid \mathbf{z}_t) \big) \propto \big\| \boldsymbol{\mu}_q(\mathbf{z}_t, \mathbf{x}) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) \big\|^2
$$

即我们只需要让模型预测的均值接近可计算的目标均值。

进一步，将 $\mathbf{x}$ 用重参数化表述为 $\mathbf{x} = \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{z}_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\varepsilon})$，代入 $\boldsymbol{\mu}_q$ 的表达式后整理：

$$
\boldsymbol{\mu}_q(\mathbf{z}_t, \mathbf{x}) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\varepsilon} \right)
$$

因此，与其让模型直接预测均值，不如让它**预测噪声 $\boldsymbol{\varepsilon}$**——这是 DDPM 最漂亮的简化：

$$
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) \right)
$$

最终的**简化训练目标**（DDPM 论文发现进一步去掉权重系数效果更好）：

$$
L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\varepsilon}} \left[ \big\| \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}\big( \sqrt{\bar{\alpha}_t} \, \mathbf{x} + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\varepsilon}, t \big) \big\|^2 \right]
$$

其中 $t \sim \text{Uniform}(1, T)$，$\mathbf{x} \sim p_{\text{data}}$，$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。**整个训练过程就是：随机取一张图片、随机选一个时间步、随机采一个噪声，让模型预测这个噪声**。

#### 训练与采样算法

**训练**（概念上的伪代码）：

1. 从数据集中采一张图 $\mathbf{x}$
2. 随机选一个扩散步 $t \in \{1, \ldots, T\}$
3. 采样噪声 $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
4. 计算 $\mathbf{z}_t = \sqrt{\bar{\alpha}_t} \mathbf{x} + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\varepsilon}$
5. 用梯度下降最小化 $\| \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) \|^2$

**采样（生成新图像）**：

1. 采样初始噪声 $\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. 从 $t = T$ 到 $1$：
   - 采样 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（$t=1$ 时不加）
   - $\mathbf{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \big( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) \big) + \sigma_t \mathbf{z}$
3. 返回 $\mathbf{z}_0$（即生成的图像）

#### 为什么 DDPM 有效：从局部到全局的渐进生成

DDPM 的成功可以类比为拼图的过程：

1. 首先你拿到一块纯色纸板（$\mathbf{z}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$）
2. 模型先 "猜" 出上面的模糊色块（大尺度结构，对应 $t \approx 500 \to 1000$ 步）
3. 然后逐渐细化轮廓和边缘（中尺度特征，对应 $t \approx 100 \to 500$ 步）
4. 最后补充纹理细节（细粒度特征，对应 $t \approx 1 \to 100$ 步）

这和艺术家作画的过程一样——先铺大色块，再勾轮廓，最后刻画细节。一次性从纯噪声跳到清晰图像几乎不可能（类比单层 VAE 只能产生模糊样本），而分成 $1000$ 小步则让每一步的任务都变得足够简单：**每步只需要去掉约 $0.1\%$ 的噪声**。

**实际应用意义**：DDPM 框架启发了 **Stable Diffusion**、**DALL·E 2**、**Imagen** 等主流图像生成模型。以 Stable Diffusion 为例，它不在像素空间直接做扩散，而是在 VAE 的隐空间中做（Latent Diffusion），大幅降低了计算开销。你在网上看到的 AI 生成的人物肖像、风景画，其背后的数学核心就是上述的噪声预测过程。

### Denoise Diffusion Implicit Models (DDIM)

DDPM 的一个主要缺点是**采样速度极慢**——要生成一张图，需要迭代 $1000$ 步，每步都要通过一次神经网络。在消费级 GPU 上生成一张 $512 \times 512$ 的图像可能需要数十秒，这严重限制了实际部署。

**问题出在哪里？** DDPM 的采样过程必须严格按照 $T$ 步的马尔可夫链进行，每一步都依赖于上一步的结果，无法并行或跳步。

**[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)** (DDIM, **Song et al.**, 2021) 发现了一个关键事实：DDPM 的训练目标 $L_{\text{simple}}$ **只依赖于 $q(\mathbf{z}_t \mid \mathbf{x})$（边缘分布），而非 $q(\mathbf{z}_{1:T} \mid \mathbf{x})$（联合分布）**。这意味着我们可以重新设计前向过程的联合分布——甚至放弃马尔可夫性——只要保持边缘分布不变，训练出的模型就依然有效。

#### 非马尔可夫前向过程

DDIM 定义了一族前向过程（参数为 $\sigma$），其联合分布为：

$$
q_\sigma(\mathbf{z}_{1:T} \mid \mathbf{x}) = q_\sigma(\mathbf{z}_T \mid \mathbf{x}) \prod_{t=2}^{T} q_\sigma(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})
$$

其中关键项 $q_\sigma(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$ 被构造为**显式依赖 $\mathbf{x}$**（因此不再满足马尔可夫性），但仍保持了边缘分布 $q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}, (1 - \bar{\alpha}_t) \mathbf{I})$ 不变。

DDIM 的逆向采样公式为：

$$
\begin{aligned}
\mathbf{z}_{t-1} &= \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{\mathbf{z}_t - \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的干净图像 } \hat{\mathbf{x}}_0} \\
&\quad + \underbrace{\sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \, \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)}_{\text{指向 } \mathbf{z}_t \text{ 的方向}} + \underbrace{\sigma_t \boldsymbol{\varepsilon}}_{\text{随机噪声}}
\end{aligned}
$$

其中 $\sigma_t$ 是一个自由参数，控制采样的随机程度。

#### DDIM 的两个极端

1. **$\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{\frac{\beta_t}{1 - \bar{\alpha}_{t-1}}}$**：**退化为 DDPM**（完全随机采样）
2. **$\sigma_t = 0$**：**完全确定性**——这就是 DDIM（"Implicit" 的含义：给定初始噪声 $\mathbf{z}_T$，生成的图像是唯一确定的，类似于隐式概率模型）

当 $\sigma_t = 0$ 时，采样公式简化为：

$$
\mathbf{z}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \hat{\mathbf{x}}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \, \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)
$$

这里没有随机噪声注入，每一步都是确定性的。

#### 加速采样

由于 DDIM 不依赖马尔可夫性，我们可以在 $\{1, \ldots, T\}$ 中选取一个**子序列** $\tau_1 < \tau_2 < \cdots < \tau_S$（$S \ll T$），直接在这些时间步上执行逆向过程而**无需重新训练模型**。例如：

- DDPM：$T = 1000$ 步
- DDIM：$S = 50$ 步（子序列均匀间隔，速度提升 $20$ 倍）

实际使用中，$S=20 \sim 50$ 步就能产生质量相当可观的样本。

#### DDIM 的隐式生成特性

确定性采样带来的一个重要特性是：**初始噪声 $\mathbf{z}_T$ 决定了生成图像的全部内容**。这类似于 GAN 中隐向量和生成图像的确定性映射，使得 DDIM 支持：

- **语义插值**：在两个初始噪声之间插值，可以得到平滑变化的图像序列。
- **属性编辑**：对 $\mathbf{z}_T$ 沿特定语义方向移动，控制生成图像的属性（类似 GAN 的隐空间操作）。
- **编码与重建**：可以用 DDIM 的逆向过程（加噪方向）将真实图像编码到噪声空间，再用正向过程（去噪方向）重建——这对图像编辑、风格迁移等下游任务至关重要。

**实际应用意义**：几乎所有部署到生产环境的扩散模型都使用了加速采样策略（DDIM、PNDM、DPM-Solver 等）。**Stable Diffusion** 默认使用 **PNDM** 采样器（DDIM 的改进版），在 $20-50$ 步内即可生成高质量图像。如果没有 DDIM 一族的技术，AI 绘图的交互式体验将不可想象——用户不可能等一分钟才看到结果。

## Score based Diffusion Models

除了从层级 VAE 和噪声预测的角度理解扩散模型，还有一条平行的数学路径——**基于分数的生成模型 (Score-based Generative Models)**。这个视角不仅提供了更深刻的数学洞察，还将 DDPM 置于一个更宏大的理论框架之中。

### 分数函数与分数匹配

对于一个概率分布 $p(\mathbf{x})$，定义其**分数函数 (Score Function)** 为对数概率密度关于数据的梯度：

$$
\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})
$$

**物理直觉**：分数函数是一个向量场，指向概率密度**增加最快的方向**。如果把 $p(\mathbf{x})$ 看作地形海拔，$\mathbf{s}(\mathbf{x})$ 就是指向山顶的箭头。

**为什么关心分数函数？** 如果我们知道 $\mathbf{s}(\mathbf{x})$，可以通过 **Langevin 动力学**从分布中采样：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \eta \, \mathbf{s}(\mathbf{x}_t) + \sqrt{2\eta} \, \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

这是一条在物理中描述粒子在势场中运动的随机微分方程：
- $\eta \, \mathbf{s}(\mathbf{x}_t)$：梯度上升 → 向高概率密度区域移动
- $\sqrt{2\eta} \, \boldsymbol{\varepsilon}$：随机扰动 → 避免卡在局部模式、保证覆盖整个分布

当 $\eta \to 0$ 且步数 $\to \infty$ 时，这些样本的分布收敛于 $p(\mathbf{x})$。

**分数匹配 (Score Matching)**：学习分数函数的方法。由于真实分数 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ 未知，我们训练模型 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$ 去逼近它，优化目标为 **Fisher 散度**（可绕过未知的归一化常数）：

$$
L_{\text{SM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \big\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x}) \big\|^2 \right]
$$

实际上，通过分部积分技巧，可以将这个目标转化为不需要知道 $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ 的形式：

$$
L_{\text{DSM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \left[ \operatorname{tr}(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})) + \frac{1}{2} \| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \|^2 \right]
$$

但在高维数据上，计算 Jacobian 的迹 ($\operatorname{tr}$) 仍然昂贵。**去噪分数匹配 (Denoising Score Matching, DSM)** 提供了一个高效的替代方案：

$$
L_{\text{DSM}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})} \left[ \big\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \big\|^2 \right]
$$

即：对数据添加已知的噪声（如高斯噪声），让模型学习去噪的分数——这等价于学习**从噪声分布恢复到数据分布的方向**。

### 多尺度噪声：NCSN

直接在高维自然图像上做分数匹配面临一个严峻挑战：**数据流形假设 (Manifold Hypothesis)**。

真实图像并非均匀填充整个像素空间——它们集中在一个极低维的流形上（把 $256 \times 256$ 的猫图随机扰动一个像素，大概率得到的不是合理的猫，而是噪声）。在这个流形之外，$\log p_{\text{data}}(\mathbf{x})$ 几乎没有定义（概率密度接近零），分数也难以估计。

解决方法来自 **[Noise Conditional Score Networks](https://arxiv.org/abs/1907.05600)** (NCSN, **Song & Ermon**, 2019)：**在不同噪声等级上分别学习分数函数**。对数据分布依次加入不同程度的高斯噪声：

$$
q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I})
$$

当 $\sigma$ 增大时，噪声将概率质量 "涂抹" 到更大的空间区域，填充了流形之间的空隙：

- $\sigma$ 小（如 $0.01$）：扰动后的数据非常接近真实图像，分数估计精确但覆盖范围有限。
- $\sigma$ 大（如 $100$）：数据被严重模糊，分数估计粗糙但能覆盖远离流形的区域。

训练一个噪声条件分数网络 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$，接收噪声等级 $\sigma$ 作为输入，学习所有尺度上的分数：

$$
L_{\text{NCSN}} = \sum_{i=1}^{L} \sigma_i^2 \, \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{q_{\sigma_i}(\tilde{\mathbf{x}} \mid \mathbf{x})} \left[ \big\| \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_i) - \nabla_{\tilde{\mathbf{x}}} \log q_{\sigma_i}(\tilde{\mathbf{x}} \mid \mathbf{x}) \big\|^2 \right]
$$

对于高斯噪声扰动，$\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) = \frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} = -\frac{\boldsymbol{\varepsilon}}{\sigma}$（$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$）。训练目标变为：

$$
L = \mathbb{E}_{\sigma, \mathbf{x}, \boldsymbol{\varepsilon}} \left[ \big\| \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x} + \sigma \boldsymbol{\varepsilon}, \sigma) + \frac{\boldsymbol{\varepsilon}}{\sigma} \big\|^2 \right]
$$

**采样 (Annealed Langevin Dynamics)**：从最大的 $\sigma$ 开始，逐步减小噪声等级，每个等级上执行若干步 Langevin 动力学。大 $\sigma$ 阶段负责 "定位" 数据的大致区域（全局结构），小 $\sigma$ 阶段负责精细化（局部细节）。

### 与 DDPM 的统一

符号可能不同，但核心数学是相通的。在 DDPM 中：

- 模型预测噪声：$\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)$
- $\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x} + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\varepsilon}$

分数视角下的对应关系为：

$$
\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) \approx \nabla_{\mathbf{z}_t} \log q(\mathbf{z}_t \mid \mathbf{x}) = -\frac{\boldsymbol{\varepsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

因此：

$$
\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) = -\sqrt{1 - \bar{\alpha}_t} \, \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{z}_t, t)
$$

**DDPM 的噪声预测网络和 NCSN 的分数网络是等价的**，仅相差一个与时间步相关的缩放因子。两者都在学习同一个目标：**给定一个被噪声污染的样本，指出 "干净" 的方向**。

### SDE 视角：连续时间的扩散

上述离散时间框架存在一个自然的连续推广，由 [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (Song et al., 2021) 给出。

**前向 SDE**（数据 → 噪声）：

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \, dt + g(t) \, d\mathbf{w}
$$

其中 $\mathbf{f}$ 是漂移系数 (drift)，$g$ 是扩散系数 (diffusion)，$\mathbf{w}$ 是维纳过程。

DDPM 对应的 SDE（VE-SDE 变体）为：

$$
d\mathbf{x} = -\frac{1}{2} \beta(t) \mathbf{x} \, dt + \sqrt{\beta(t)} \, d\mathbf{w}
$$

NCSN 对应的 SDE（VE-SDE）为：

$$
d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}} \, d\mathbf{w}
$$

**逆向 SDE**（噪声 → 数据），Anderson (1982) 的结果给出：

$$
d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g(t) \, d\bar{\mathbf{w}}
$$

其中 $\bar{\mathbf{w}}$ 是逆向时间的维纳过程。**请注意这个形式的美**：逆向过程的形式和前向完全对称，唯一的区别是将 $\mathbf{f}$ 替换为 $\mathbf{f} - g^2 \nabla_{\mathbf{x}} \log p_t$。如果我们学习到了分数函数 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，就可以直接代入运行逆向 SDE 来生成样本。

SDE 框架的意义在于：
1. **统一性**：DDPM 和 NCSN 都是同一 SDE 的不同离散化方案。
2. **ODE 采样**：每个 SDE 对应一个等价的概率流 ODE（Probability Flow ODE），可以用确定性的 ODE 求解器高效生成样本。
3. **理论优雅**：SDE 的形式自然连接了扩散模型与统计物理、随机过程、最优控制等领域的经典理论。

### 一个统一的故事线

现在让我们把整条逻辑链串起来：

1. 我们想生成图像 → 需要学习数据分布 $p(\mathbf{x})$
2. 直接建模高维分布太难 → 引入隐变量 → **VAE**
3. 单层隐变量表达能力不足 → 引入多层隐变量 → **HVAE**
4. 训练深层 HVAE 太难 → 固定编码器为加噪过程 → **DDPM**
5. 采样太慢 → 放弃马尔可夫性 → **DDIM**
6. 换个角度看，DDPM 的噪声预测本质上是在学习分数函数 → **Score-based Models**
7. 连续极限 → 随机微分方程 → **SDE 框架**

每一步都解决了一个具体的问题，而每一步的解决方案又在数学上优雅地嵌套在前一步的框架内。这正是扩散模型研究的美感所在。

### 实际应用全景

扩散模型的影响力远远超出了学术论文：

- **文生图 (Text-to-Image)**：**Stable Diffusion**、**DALL·E 2/3**、**Midjourney**、**Imagen** 都以扩散模型为核心。用户输入 "一只戴着墨镜的柴犬在沙滩上冲浪"，模型在隐空间中执行逆向扩散过程，由文本条件引导生成方向。

- **图像编辑**：**InstructPix2Pix**、**ControlNet** 利用扩散模型的条件生成能力和 DDIM 的编码-重建特性，实现了局部编辑、风格迁移、姿态控制等功能。

- **3D 生成**：**DreamFusion** (Poole et al., 2022) 使用预训练的 2D 扩散模型作为先验，通过 Score Distillation Sampling (SDS) 优化 3D 表示（NeRF），实现了 "从文字描述到 3D 模型" 的生成。

- **分子与材料设计**：扩散模型被用于生成分子的 3D 构象、蛋白质结构和晶体材料结构——这些领域的 "图像" 变成了原子坐标和化学键。

- **视频生成**：**Sora** (OpenAI)、**Runway Gen-3** 等视频生成模型也基于扩散架构，在时空潜变量空间中进行去噪，生成连续的视频帧。

- **音频与语音合成**：**WaveGrad**、**DiffWave** 等将扩散模型应用于原始波形生成，实现了比传统声码器更高的音质。

这些应用的本质都归功于扩散模型的一个核心优势：**训练稳定、生成质量高（超过了 GAN），且不像 GAN 那样容易出现模式坍缩 (mode collapse)**。

---
*本文参考了 [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) (Calvin Luo, 2022)、[DDPM](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)、[DDIM](https://arxiv.org/abs/2010.02502) (Song et al., 2021)、[NCSN](https://arxiv.org/abs/1907.05600) (Song & Ermon, 2019)、[Score-Based SDE](https://arxiv.org/abs/2011.13456) (Song et al., 2021) 等论文。*

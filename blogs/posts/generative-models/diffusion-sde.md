# Diffusion Model 的 SDE 表示

我们已经习惯了用**隐空间概率分布**的方式理解扩散模型的原理：

- 从 `HVAE` 出发，我们用固定的前向高斯分布将图像映射到隐空间中，扩散模型实际上做的是学习求解**逆向过程的分布参数**
- 优化目标也是通过 `ELBO` 推导出来的，用这种方法训练的模型学习的是**估计逆向的噪声**

下面我们再介绍一种理解扩散模型的角度，从随机微分方程的角度理解扩散模型的原理。

扩散模型的每一步前向过程其实都是在求解**离散化的随机微分方程**，模型学习的是如何求解这个随机微分方程

用 `SDE` 描述一个“加噪过程”：$dx_t=u(x_t, t)dt+\sigma_tdW$。

其中 `W` 服从一个维纳过程，即 $dW\sim\mathcal{N}(0,t\cdot I)$。

离散化后可得：$x_{t+\Delta t}=x_t+u(x_t,t)\Delta t+\sigma_t\sqrt{\Delta t}\cdot\epsilon,\epsilon\sim\mathcal{N}(0,I)$。

对任意一个测试函数 $\phi(x)$，有：

$$
\begin{aligned}
    \phi(x_{t+\Delta t})&=\phi(x_t+u(x_t,t)\Delta t+\sigma_t\sqrt{\Delta t}\cdot\epsilon) \\
    &=\phi(x_t)+\big(u(x_t,t)\Delta t+\sigma_t\sqrt{\Delta t}\cdot\epsilon\big)\cdot\nabla_{x_t}\phi(x_t)+\frac{1}{2}\sigma_t^2\nabla_{x_t}^2\phi(x_t)\cdot\epsilon^2\Delta t+o(\Delta t) \\
    &\approx \phi(x_t)+\big(u(x_t,t)\Delta t+\sigma_t\sqrt{\Delta t}\cdot\epsilon\big)\cdot\nabla_{x_t}\phi(x_t)+\frac{1}{2}\sigma_t^2\nabla_{x_t}^2\phi(x_t)\cdot\epsilon^2\Delta t
\end{aligned}
$$

根据 $\mathbb{E}[\epsilon]=0,\mathbb{E}[\epsilon^2]=1$，化简期望可得：

$$
\begin{aligned}
    \mathbb{E}\big[\phi(x_{t+\Delta t})\big]&\approx\mathbb{E}\big[\phi(x_t)\big]+\mathbb{E}\big[u(x_t,t)\Delta t\cdot\nabla_{x_t}\phi(x_t)\big]+\mathbb{E}\big[\frac{1}{2}\sigma_t^2\nabla_{x_t}^2\phi(x_t)\Delta t\big]
\end{aligned}
$$

将期望展开：

$$
\begin{aligned}
\int\phi(x_{t+\Delta t})\cdot p_{t+\Delta t}(x_{t+\Delta t})dx_{t+\Delta t}
&=\int\left(\phi(x_t)+\big(u(x_t,t)\nabla_{x_t}\phi(x_t)+\frac{1}{2}\sigma_t^2\nabla_{x_t}^2\phi(x_t)\big)\Delta t\right)\cdot p_{t}(x_t)dx_t
\end{aligned}
$$

变量符号不改变积分结果：

$$
\int\phi(x)\cdot p_{t+\Delta t}(x)dx =
\int\left(\phi(x)+\big(u(x,t)\nabla_{x}\phi(x)+\frac{1}{2}\sigma_t^2\nabla_{x}^2\phi(x)\big)\Delta t\right)\cdot p_{t}(x)dx
$$

移项整理可得：

$$
\begin{aligned}
    \int\phi(x)\cdot\frac{p_{t+\Delta t}(x)-p_{t}(x)}{\Delta t}dx=\int\big(u(x,t)\nabla_{x}\phi(x)+\frac{1}{2}\sigma_t^2\nabla_{x}\cdot\nabla_{x}\phi(x)\big)\cdot p_t(x)dx
\end{aligned}
$$

根据**分部积分**法则：

$$
\begin{aligned}
(uv)'&=u'v+uv' \\
\Rightarrow \int u'vdx&=uv-\int uv'dx \\
\Rightarrow \int u'vdx&=-\int uv'dx\quad\text{if }uv|_{-\inf}^{+\inf}=0
\end{aligned}
$$

扩展到多元的形式依然有同样的结论：

$$
\begin{aligned}
\int\mathbf{v}\cdot\nabla udx&=-\int u\cdot\nabla\mathbf{v}dx\quad \text{if }u\cdot\nabla v|_{-\inf}^{+\inf}=0 \\
\int\mathbf{v}\cdot\nabla^2 udx&=\int u\cdot\nabla^2\mathbf{v}dx\quad \text{if }\nabla u\cdot\nabla v|_{-\inf}^{+\inf}=0
\end{aligned}
$$

显然对于一般的概率密度函数有以下结论成立：

$$
p(x)\rightarrow 0,\quad \nabla_x p(x)\rightarrow 0,\quad x\rightarrow \pm\inf
$$

于是（5）式可以整理为：

$$
\int\phi(x)\frac{\partial p_t(x)}{\partial t}dx=\int\left(-\nabla_x\big(u(x,t)\cdot p_t(x)\big)+\frac{1}{2}\sigma_t^2\nabla_x^2 p_t(x)\right)\cdot\phi(x)dx
$$

根据测试函数法，可以得到 **`Fokker-Planck` 方程**：

$$
\frac{\partial p_t(x)}{\partial t}=-\nabla_x\big(u(x,t)\cdot p_t(x)\big)+\frac{1}{2}\sigma_t^2\nabla_x^2 p_t(x)
$$

若将时间反向，令 $\tau=T-t$，

T 为终止时间，那么我们可以推导**逆向的 `Fokker-Planck` 方程**

证明的核心思想非常巧妙：**既然前向 SDE 和前向 Fokker-Planck (FP) 方程是一一对应的，那么我们只需要把前向的 FP 方程“时光倒流”，看看它对应着怎样的一个新的 SDE 就可以了**

为了清晰起见，我们将标量化处理，并假设扩散系数 $g(t)$ 仅依赖于时间。

已知**前向 SDE** 为：$dx_t = f(x_t, t)dt + g(t)dW_t$。

它对应的概率密度记为 $p_t(x)$，

满足前向 FP 方程：

$$
\frac{\partial p_t}{\partial t} = -\nabla \cdot (f p_t) + \frac{1}{2}g^2 \nabla^2 p_t
$$

我们想看时间倒流时会发生什么，定义一个反向时间变量 $\tau = T - t$，其中 $T$ 是总时间。

那么，当正向时间 $t$ 增加时，反向时间 $\tau$ 也在增加。由于 $d\tau = -dt$，

我们对概率密度求偏导时，必然有：

$$
\frac{\partial p}{\partial \tau} = -\frac{\partial p}{\partial t}
$$

把这个关系代入第一步的前向 FP 方程，我们在等式两边同乘 $-1$，

得到**反向时间的偏微分方程**：

$$
\frac{\partial p}{\partial \tau} = \nabla \cdot (f p_t) - \frac{1}{2}g^2 \nabla^2 p_t
$$

我们的目标是：在这个反向时间 $\tau$ 里，寻找一个新的 SDE（即逆向 SDE）：

$$
d\bar{x}_\tau = \tilde{f}(\bar{x}_\tau, \tau)d\tau + \tilde{g}(\tau)d\bar{W}_\tau
$$

这个新的 SDE，必然也有**属于它自己的、标准的 FP 方程**：

$$
\frac{\partial p}{\partial \tau} = -\nabla \cdot (\tilde{f} p_t) + \frac{1}{2}\tilde{g}^2 \nabla^2 p_t
$$

我们回到通过直接逆向得到的方程：

$$
\frac{\partial p}{\partial \tau} = \nabla \cdot (f p_t) - \frac{1}{2}g^2 \nabla^2 p_t
$$

为了让它的后半部分出现目标方程里的 $\boldsymbol{+\frac{1}{2}g^2 \nabla^2 p_t}$，

我们运用“加一项减一项”技巧

在等式右边同时 **减去** $g^2 \nabla^2 p_t$，并 **加上** $g^2 \nabla^2 p_t$，

得到：

$$
\frac{\partial p}{\partial \tau} = \nabla \cdot (f p_t) - g^2 \nabla^2 p_t + \frac{1}{2}g^2 \nabla^2 p_t
$$

现在，把前两项合并，提取出散度算子 $\nabla \cdot$。

因为 $\nabla^2 p_t = \nabla \cdot \nabla p_t$，

所以有：

$$
\frac{\partial p}{\partial \tau} = \nabla \cdot (f p_t - g^2 \nabla p_t) + \frac{1}{2}g^2 \nabla^2 p_t
$$

接着，使用一个极其关键的微积分恒等式。对数求导法则为 $\nabla p_t = p_t \nabla \log p_t$。

再把它替换掉上式括号里的 $\nabla p_t$，

得到：

$$
\frac{\partial p}{\partial \tau} = \nabla \cdot (f p_t - g^2 p_t \nabla \log p_t) + \frac{1}{2}g^2 \nabla^2 p_t
$$

最后，把括号里的 $p_t$ 提出来，并在最前面提出一个负号，以完全匹配 FP 方程的标准格式：

$$
\frac{\partial p}{\partial \tau} = -\nabla \cdot \left( \left[-f + g^2 \nabla \log p_t\right] p_t \right) + \frac{1}{2}g^2 \nabla^2 p_t
$$

现在，把上面这个变形后的方程，和我们的目标方程对比：

$$
\frac{\partial p}{\partial \tau} = -\nabla \cdot ( \tilde{f} p_t ) + \frac{1}{2} \tilde{g}^2 \nabla^2 p_t
$$

对比系数，答案呼之欲出：
1. **新的扩散系数：**

   $$\tilde{g}^2 = g^2$$

   即 $\tilde{g} = g$。

   扩散噪声的大小不变。

2. **新的漂移系数：**

   $$\tilde{f} = -f + g^2 \nabla \log p_t$$

把它们代回反向时间的 SDE 中，我们就得到了关于反向时间 $\tau$ 的方程：

$$
d\bar{x}_\tau = \left[-f + g^2 \nabla \log p_t\right] d\tau + g d\bar{W}_\tau
$$

在生成式 AI 领域的论文中，为了方便，大家习惯继续使用物理时间 $t$。此时 $dt$ 代表一个负的时间增量，即时光倒流，同时 $d\bar{W}_t=\sqrt{|t|}\epsilon$，

所以公式里的 $f$ 前面的负号会被抵消，最终写成经典逆向 SDE 形式：

$$
dx_t = \left[f(x_t, t) - g^2(t) \nabla_x \log p_t(x_t)\right] dt + g(t) d\bar{W}_t
$$

可以训练一个模型学习求解这个逆向过程，训练目标显然是估计得分函数 $\nabla_x\log p_t(x_t)$，

这样我们只需做数值积分就能得到初始的结果

---
title: 'Transform (Continue)'
date: 2025-07-06
permalink: /notes/2025/07/transform-cont/
category: 'notes'
---

回顾一下计算机图形学需要完成的任务：

在实际生产中我们可以通过一些手段获得表示三维模型的点云，我们需要通过合适的方法将三维模型按照符合人眼视觉的形式绘制到二维屏幕上

<!--more-->

类比拍照的三个步骤，我们绘制的过程也分为三步：

1. Model Transformation：摆好姿势
2. View Transformation：架好相机
3. Projection Transformation：按下快门

简称 `MVP`，其中 `M` 和 `V` 都需要使用之前介绍的变换方法进行处理，而 `P` 则是本文将要着重介绍的投影变换

## View Transformation

我们只关心模型和相机的相对位置，因此我们可以将模型和相机作为整体，一起平移至一个方便研究的位置，这个过程称为视图变换 (View Transformation)

约定默认的相机位姿是：位于原点(0, 0, 0)，沿着 -z 轴的方向看（这是右手系的特点）$$\hat{g} = (0, 0, -1)$$，向上方向是 y 轴的正方向（相机不能倾斜）$$\hat{t} = (0, 1, 0)$$

![axis]({{ site.baseurl }}/images/right_hand_axis.png)

设相机的初始位置为 $$\vec{e} = (x_e, y_e, z_e)$$，初始方向为 $$\vec{g} = (g_x, g_y, g_z)$$，初始的向上方向为 $$\vec{t} = (x_t, y_t, z_t)$$

需要注意的是，为了保持相机和模型的相对位置不变，在改变相机位姿的同时，模型也需要跟着改变

先使用平移变换将相机平移至原点： $$T_{view} = \begin{pmatrix}
1 & 0 & 0 & -x_e \\
0 & 1 & 0 & -y_e \\
0 & 1 & 0 & -z_e \\
0 & 1 & 0 & 1 \\
\end{pmatrix}$$

如何将 $$\vec{g} \& \vec{t}$$ 旋转至 -z 和 +y 方向？

考虑**逆过程** —— 将 x 轴，-z 轴，和 y 轴方向旋转至 $$\vec{g} \times \vec{t}, \vec{g}, \vec{t}$$ （注意使用的是右手系，左手系略有不同）

$$R_{view}^{-1}\cdot \begin{pmatrix}
1 \\
0 \\
0 \\
0 \\
\end{pmatrix} = \begin{pmatrix}
x_{\vec{g} \times \vec{t}} \\
y_{\vec{g} \times \vec{t}} \\
z_{\vec{g} \times \vec{t}} \\
0 \\
\end{pmatrix}$$

$$R_{view}^{-1}\cdot \begin{pmatrix}
0 \\
0 \\
-1 \\
0 \\
\end{pmatrix} = \begin{pmatrix}
x_{\vec{g}} \\
y_{\vec{g}} \\
z_{\vec{g}} \\
0 \\
\end{pmatrix}$$

$$R_{view}^{-1}\cdot \begin{pmatrix}
0 \\
1 \\
0 \\
0 \\
\end{pmatrix} = \begin{pmatrix}
x_{\vec{t}} \\
y_{\vec{t}} \\
z_{\vec{t}} \\
0 \\
\end{pmatrix}$$

可知 $$R_{view}^{-1} = \begin{pmatrix}
x_{\vec{g} \times \vec{t}} & x_{\vec{t}} & -x_{\vec{g}} & 0\\
y_{\vec{g} \times \vec{t}} & y_{\vec{t}} & -y_{\vec{g}} & 0\\
z_{\vec{g} \times \vec{t}} & z_{\vec{t}} & -z_{\vec{g}} & 0\\
0 & 0 & 0 & 1
\end{pmatrix}$$

由旋转矩阵的正交性：$$R_{view}=(R_{view}^{-1})^T$$

至此，我们已经成功地将相机位姿调整至合适的位置，接下来需要做的是使用投影变换将三维模型投影到屏幕上

## Projection Transformation

投影变换主要分为两种：

- Orthographic Projection (正交投影)：三维世界中平行的线，投影后依然平行（不会发生“近大远小”的现象，多用于工程制图）
- Perspective Projection (透视投影)：三维世界中平行的线，投影后延长线相交（会发生“近大远小”的现象，符合人眼视觉）

### 正交投影

只使用仿射变换，将模型和相机的 “立方体包围盒” 投影到标准立方体: $$[-1, 1] \times [-1, 1] \times[-1, 1]$$

设初始包围盒为：$$[l, r] \times [b, t] \times [f, n]$$ （分别表示左右，上下，远近，由于向着 -z 看，越远坐标值越小）

那么 $$M_{ortho} = \begin{pmatrix}
\frac{2}{r-l} & 0 & 0 & 0 \\
0 & \frac{2}{t-b} & 0 & 0 \\
0 & 0 & \frac{2}{n-f} & 0 \\
0 & 0 & 0 & 1 \\
\end{pmatrix} \cdot \begin{pmatrix}
1 & 0 & 0 & -\frac{l+r}{2} \\
0 & 1 & 0 & -\frac{t+b}{2} \\
0 & 0 & 1 & -\frac{f+n}{2} \\
0 & 0 & 0 & 1 \\
\end{pmatrix}$$

### 透视投影

思路是先将透视投影转化为正交投影，再用正交投影的方式进行投影

使用的一个假设：

- 相机视角范围边界上的点经过透视-正交转换（$$M_{perspe->ortho}$$）后会位于立方体的棱边上

![perspe->ortho]({{ site.baseurl }}/images/perspe2ortho.png)

有两条重要性质：

- 距离相机最近的平面（平行于 xOy）上的点，经过透视-正交转换后**坐标不变**
- 距离相机最远的平面（平行于 xOy）上的点，经过透视-正交转换后 **z 坐标不变**

根据相似三角形的比例关系可以得到变换后点的 x, y 坐标，但是除了两侧的点外，其余点的 z 坐标在透视-正交变换下会发生偏移，是一个未知的参数

![perspe->ortho]({{ site.baseurl }}/images/perspe_xy.png)

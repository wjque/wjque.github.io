---
title: 'Transform'
date: 2025-07-04
permalink: /notes/2025/07/transform/
category: 'notes'
---

3维世界中的物体称为 `model`，所谓 Animation 就是 `model` 的 scaling。

<!--more-->

区别于计算机视觉由 2D 信息复原 3D 信息的任务，计算机图形学关心的是如何将 3D 的 `model` 呈现在屏幕 `screen` 上。

而三维世界中的 `model` 常用一定数量的三角形表示，并且三角形的边通常使用顶点之间的连接关系来表示，因此我们通常处理的都是三维世界中的顶点集，故顶点的变换法则显得尤为重要，下面先介绍二维世界的变换法则，由此推及到三维的情形。

## 2D Transfroms

研究变换的矩阵形式，关键在考虑特殊点的坐标变换，下面的推导过程都运用了这一思想

### Scale (缩放)

对二维图像进行缩放操作，等价于对面内的每一个顶点 $$(x, y)^T$$ 进行相同的比例变换，假设横向缩放 $$s_x$$ 倍，纵向缩放 $$s_y$$ 倍，那么缩放矩阵可以表示为： $$S_{x,y} = \begin{pmatrix}
{s_x} & 0 \\
0 & {s_y}
\end{pmatrix}$$

![Scale]({{ site.baseurl }}/images/Scale.png)

### Reflection (反射)

考虑关于坐标轴的反射，关于 y 轴的反射只需将横坐标变为原来的相反数，纵坐标不变即可，关于 x 轴的反射同理，以关于 y 轴的反射为例，反射矩阵科研表示为： $$\begin{pmatrix}
-1 & 0 \\
0 & 1 
\end{pmatrix}$$ 

### Shear (切变)

直观上可以理解四边形在横向和纵向有一定的“弹性”，可以横向、纵向拉伸

![Shear]({{ site.baseurl }}/images/Shear.png)

假设四边形沿着 x 轴发生了长度为 a 的切变，考虑左上角顶点 (0, 1) 和左下角顶点 (0, 0) 的变化，横坐标等价于加上了与纵坐标相关的特定值，可以提炼出切变的矩阵形式： $$\begin{pmatrix}
1 & a \\
0 & 1 \\
\end{pmatrix}$$

### Rotate (旋转)

在无特殊说明的情况下，都是绕原点逆时针旋转

![Rotate]({{ site.baseurl }}/images/Rotate.png)

同样使用特殊点来推导：$$(0, 1) \Rightarrow (-sin \theta, cos \theta),\quad (1, 0) \Rightarrow (cos \theta,, sin \theta)$$，于是旋转 $$\theta$$ 角的矩阵表示为： $$R_{\theta} = \begin{pmatrix}
cos \theta & -sin \theta \\
sin \theta & cos \theta \\
\end{pmatrix}$$

### Translation (平移)

前面的几种变换都可以使用一个矩阵表示，它们统称为 **线性变换**

然而平移变换这一简单的变换却不能使用一个矩阵表示，因此它不是线性变换： $$\begin{pmatrix}
x' \\
y' \\
\end{pmatrix} = \begin{pmatrix}
1 & 0 \\
0 & 1 \\
\end{pmatrix}\ \begin{pmatrix}
x \\
y \\
\end{pmatrix} + \begin{pmatrix}
t_x \\
t_y \\
\end{pmatrix}$$

![Translation]({{ site.baseurl }}/images/Translation.png)

### Homogeneous Coordinate (齐次坐标)

我们希望使用统一的矩阵表达来表示上述所有变换，尤其是平移变换。由于二维的平移变换不能使用一个二阶矩阵表示，故我们引入齐次坐标，增加一个维度：$$2D\ Point = (x, y, 1)^T,\ 2D\ Vector = (x, y, 0)^T$$，那么二维点的平移变换可以表示为： $$\begin{pmatrix}
x' \\
y' \\
w' \\
\end{pmatrix}=\begin{pmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1 \\
\end{pmatrix}\ \begin{pmatrix}
x \\
y \\
1 \\
\end{pmatrix} = \begin{pmatrix}
x + t_x \\
y + t_y \\
1 \\
\end{pmatrix}$$

我们在第三个维度中使用 0 和 1 来区别 “二维向量” 和 “二维点”，下面说明这样规定的合理性：

由于 vector 的第三个维度为0，所以：vector + vector = vector, vector - vector = vector

point的第三个维度为1，所以：point - point = vector

那么 point + point = ?

规定： $$w\not= 0,\ \begin{pmatrix}
x \\
y \\
w \\
\end{pmatrix}\quad represents \quad 2D\ point:\begin{pmatrix}
x/w \\
y/w \\
1
\end{pmatrix}$$，所以 point + point 表示的实际上是二者的中点 

可表示为以下形式的变换统称为**仿射变换（Affine Transform）**： $$\begin{pmatrix}
x' \\
y' \\
\end{pmatrix}=\begin{pmatrix}
a & b \\
c & d \\
\end{pmatrix}\ \begin{pmatrix}
x \\
y \\
\end{pmatrix} + \begin{pmatrix}
t_x \\
t_y \\
\end{pmatrix}$$

仿射变换的共同点是**最后一行都为(0, 0 , 1)**，需要注意的是未来将要经常使用的投影变换（Project Transform）不属于仿射变换

缩放变换矩阵的齐次坐标表示： $$S_{x,y}=\begin{pmatrix}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & 1 \\
\end{pmatrix}$$

旋转变换矩阵的齐次坐标表示： $$R_{\theta}=\begin{pmatrix}
cos \theta & -sin \theta & 0 \\
sin \theta & cos \theta & 0 \\
0 & 0 & 1 \\
\end{pmatrix}$$

平移变换矩阵的齐次坐标表示： $$S_{x,y}=\begin{pmatrix}
1 & 0 & t_x \\
0 & 1 & t_y \\
0 & 0 & 1 \\
\end{pmatrix}$$

## 3D Transforms

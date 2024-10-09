---
layout: post
title:  Introduction to torch.autograd.backward() 
date:   2024-10-08 10:29:33 +0000
categories: jekyll update
---

我们在尝试构建基于pipeline的模型训练过程中，发现一个函数：`torch.autograd.backward()`，咋一看跟理解中什么反向传播，链式法则都关联不起来。基于本人记性比较差，在这里特地记录一下这个函数的作用和用法。在介绍此函数之前，首先使用一个简单的例子介绍一下反向传播的过程。

假设我们有一个tensor $\mathbf x$，其定义如下：

$$
\mathbf x =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

然后，我们有如下定义的函数$f(\mathbf x)$:

$$
f(\mathbf x) = \mathbf x^2 =
\begin{bmatrix}
x_1^2 \\
x_2^2
\end{bmatrix}
$$

且我们有另外一个函数$l(\mathbf x)$定义如下：

$$
l(\mathbf x) =
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
= x_1 + x_2
$$

根据上面的定义，假设有一个tensor $\mathbf z = \begin{bmatrix} z_1 \\ z_2 \end{bmatrix}$以及如下的计算过程：

$$
\begin{align*}
\mathbf y &= f(\mathbf z) \qquad (1) \\
\mathbf \delta &= l(\mathbf y) \qquad (2)
\end{align*}
$$

如果这个时候，我们要求$\frac{\partial \delta}{ \partial \mathbf z}$,显然我们可以使用下面的链式法则：

$$
\begin{align*}
\frac{\partial \delta}{ \partial \mathbf z} &= \frac{ \partial \delta }{ \partial \mathbf y} \cdot \frac{\partial \mathbf y}{ \partial \mathbf z} \\
&= \begin{bmatrix}
\frac{\partial y_1}{\partial z_1} & \frac{\partial y_1}{\partial z_2} \\
\frac{\partial y_2}{\partial z_1} & \frac{\partial y_2}{\partial z_2}
\end{bmatrix}^T

\cdot

\begin{bmatrix}
\frac{\partial \delta}{\partial y_1} \\
\frac{\partial \delta}{\partial y_2}
\end{bmatrix} \\

&= \begin{bmatrix}
\frac{\partial y_1}{\partial z_1} & \frac{\partial y_2}{\partial z_1} \\
\frac{\partial y_1}{\partial z_2} & \frac{\partial y_2}{\partial z_2}
\end{bmatrix}

\cdot

\begin{bmatrix}
\frac{\partial \delta}{\partial y_1} \\
\frac{\partial \delta}{\partial y_2}
\end{bmatrix} \\

&= \begin{bmatrix}
\frac{\partial \delta}{\partial y_1} \cdot \frac{\partial y_1}{\partial z_1} \\
\frac{\partial \delta}{\partial y_2} \cdot \frac{\partial y_2}{\partial z_2}
\end{bmatrix}

\end{align*} \qquad (3)
$$

其中，假设我们定义Jacobian矩阵如下：

$$
\begin{align*}
\mathbf J &= \left [  \frac{\partial \mathbf y}{ \partial z_1} \quad \frac{\partial \mathbf y}{\partial z_2}   \right] \\

&= \begin{bmatrix}
\frac{\partial y_1}{ \partial z_1} & \frac{\partial y_1}{ \partial z_2} \\
\frac{\partial y_2}{ \partial z_1} & \frac{\partial y_2}{ \partial z_2}
\end{bmatrix}
\end{align*} \qquad (4)
$$

将式子$(4)$代入到$(3)$中，我们可以得到：

$$
\begin{align*}
\frac{\partial \delta}{ \partial \mathbf z} &=  \frac{ \partial \delta }{ \partial \mathbf y} \cdot \frac{\partial \mathbf y}{ \partial \mathbf z} \\

&= \mathbf J^T \cdot \frac{\partial \mathbf \delta}{ \partial \mathbf y }
\end{align*} \qquad (5)
$$

介绍了上面的一大堆$\delta$对$\mathbf z$的梯度的计算过程，下面我们来看如何使用pytorch中的函数`torch.autograd.backward()`来实现上述的计算过程，这个函数中最重要的参数有三个

* 第一、tensors，这个参数实际上对应公式$(5)$中的$\mathbf y$，官方文档中说明如下

```text
tensors (Sequence[Tensor] or Tensor) – Tensors of which the derivative will be computed.
```

* 第二、grad_tensors，这个参数对应公式$(5)$中的$\frac{\partial \delta }{ \partial \mathbf y}$，官方文档说明如下

```text
grad_tensors (Sequence[Tensor or None] or Tensor, optional) – The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional.
```

* 第三、inputs，这个参数对应公式$(5)$中的$\mathbf z$，这个参数是可选的，主要用来说明在计算图中，哪些叶节点（leaf node）是我们真正关心并想要求解其梯度的。官方文档说明：

```text
inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge], optional) – Inputs w.r.t. which the gradient be will accumulated into .grad. All other Tensors will be ignored. If not provided, the gradient is accumulated into all the leaf Tensors that were used to compute the tensors.
```

下面我们代入一个具体的例子，具体看看会发生什么

```python
import torch
z = torch.tensor([1.0, 3.0], requires_grad=True)
y = z**2
l = y.sum()
print(z)
print(y)
print(l)
```

执行后得到如下结果：

```python
tensor([1., 3.], requires_grad=True)
tensor([1., 9.], grad_fn=<PowBackward0>)
tensor(10., grad_fn=<SumBackward0>)
```

显然根据上面的例子：

$$
\frac{\partial l}{ \partial \mathbf y} =
\begin{bmatrix}
1.0 & 1.0
\end{bmatrix}
$$

所以：

```python
gradient_tensors = torch.tensor([1.0, 1.0])
torch.autograd.backward(tensors = y, grad_tensors = gradient_tensors)
print(z.grad)
```

结果为：

```
tensor([2., 6.])
```

如果我们直接从$l$处计算关于$\mathbf z_1$的梯度，我们又会得到什么值？让我们来看下

```
import torch
z1 = torch.tensor([1.0, 3.0], requires_grad=True)
y1 = z1**2
l = y1.sum()
l.backward()
print(z1.grad)
```

同样的，我们可以得到`z1.grad`的值为：

```python
tensor([2., 6.])
```



参考文献：

* https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
* https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC
* https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
* https://medium.com/@saihimalallu/how-exactly-does-torch-autograd-backward-work-f0a671556dc4
* deepspeed/deepspeed/runtime/pipe/engine.py:851
* 这里有pipeline engine对于相关部分的实现代码；
* https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
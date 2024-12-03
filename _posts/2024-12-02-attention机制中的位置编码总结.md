---
layout: post
title:  Attention机制中的位置编码总结 
date:   2024-12-02 10:29:33 +0000
categories: jekyll update
---

Position embedding（位置嵌入）最早来自于transformer的原始论文。也是因为有了attention机制，所以才需要有position embedding 。position embedding是对一个序列中的每一个token的位置进行编码，使得位置信息也作为特征被加入到attention的计算中去。

假设我们有一个词嵌入的序列：$\mathbf X =  \mathbf x^0, \mathbf x^1, ... \mathbf x^{n-1} $， $\mathbf x^i$ 是序列中第 $i$个token的词嵌入，$\mathbf x^i \in \mathbb R^h$ (其中$h$是词嵌入的维度数)。每一个$\mathbf x^i$对应的位置$i$的position embedding我们使用$\mathbf p^i$来表示，一般来说$\mathbf p^i$和$\mathbf x^i$具有相同的维度数，即$\mathbf p^i \in \mathbb R^h$；这样方便两者结合，例如最简单的结合方式是$\mathbf p^i$跟$\mathbf x^i$的逐位相加，即$\mathbf p^i_j + \mathbf x^i_j$)【说明：本文全文中，对于位置嵌入，以及词嵌入，一般使用**上标**表示该token在序列中的位置；使用**下标**表示在embedding中的维度；例如$\mathbf p^i_j \in \mathbb R$ 表示在序列中的第$i$个位置嵌入的第$j$个维度上的取值】

在transformer论文中，$\mathbf p^i$与词嵌入$\mathbf x^i$ 使用下面的计算公式被加入到attention的计算中：

$$
f_{t:t \in \{q,k,v\}}(\mathbf x^i, i) = \mathbf W_{t:t\in \{q,k,v\}}(\mathbf x^i + \mathbf p^i) \qquad (0.1)
$$

即attention机制中，在位置$i$上的query, key, value可分别表示为$\mathbf q_i, \mathbf k_i, \mathbf v_i$ (注意：在q,k,v的表达中，我们使用**下标**表示其在序列中对应的位置)：

$$
\begin{align*}
\mathbf q_i &= \mathbf W_q(\mathbf x^i + \mathbf p^i) \\
\mathbf k_i &= \mathbf W_k(\mathbf x^i + \mathbf p^i) \\
\mathbf v_i &= \mathbf W_v(\mathbf x^i + \mathbf p^i)
\end{align*} \qquad (0.2)
$$

对于$\mathbf q_i$对$\mathbf k_j$对应的权重输出以及则可以表示为$\mathbf a_{i,j}$：

$$
\mathbf a_{i,j} = \frac{exp(\frac{\mathbf q_i \mathbf k_j}{\sqrt h})}{\sum_{s=0}^{n-1} exp( \frac{\mathbf q_i \mathbf k_s}{\sqrt h})} \qquad (0.3)
$$

对于$\mathbf q_i$对应的attention的计算输出值$\mathbf o_i$则可以表示为：

$$
\mathbf o_i = \sum_{j=0}^{n-1} \mathbf a_{i,j} \mathbf v_j \qquad (0.4)
$$

看明白了$\mathbf p^i$是如何在attention机制中使用，我们再来看看$\mathbf p^i$是如何被定义的。

# 1. Sinusoidal Position Embedding

我们首先来看一下transformer原始论文中的position embedding （Sinusoidal Position Embedding）(基于正弦曲线的位置嵌入)是如何实现我们上面定义的$\mathbf p^i$。

## 1.1 定义

我们定义$\mathbf p^i_j$  ($ 0 \le j \lt h$) 如下面的公式（1.1）

$$
\mathbf p^i_j = 
\begin{cases}
sin(\frac{i}{10000^{\frac{2t}{h}}}) \qquad &\text{if}\:j = 2t  \\
cos(\frac{i}{10000^{\frac{2t}{h}}}) \qquad &\text{if}\:j = 2t+1
\end{cases} \qquad (1.1)
$$

这个公式乍一看非常的抽象，想要解释它的由来也有种千头万绪的感觉。我们从学习数学的角度上来说，第一次见到一个不了解的概念，最好的学习方法其实是不去深究这个到底是什么，而是看看这个东西具体能干什么。

根据[Linear Relationships in the Transformer’s Positional Encoding](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/)，我们也可以把这里的$\mathbf p^i$写成下面的形式

$$
\mathbf p^i = \begin{bmatrix}
\sin(\frac{i}{f_1}) \\
\cos(\frac{i}{f_1}) \\
\sin(\frac{i}{f_2}) \\
\cos(\frac{i}{f_2}) \\
\vdots \\
\sin(\frac{i}{f_{\frac{h}{2}}}) \\
\cos(\frac{i}{f_{\frac{h}{2}}}) \\
\end{bmatrix} \qquad (1.2)
$$

其中，

$$
f_t := 10000^{\frac{2t}{h}} := \frac{1}{\lambda_t} \quad (1.3)
$$

且$t = 0,1,...\frac{h}{2}$  ，这种写法把论文中的原始公式进行了形式上的调整，引入了频率以及波长的概念（我们在中学课本上，一般使用字母$f$表示频率，而使用字符$\lambda$表示波长）

从定义(1.3)中，得到$ \lambda_t = \frac{1}{f_t}$，并代入到公式（1.2）中得到：

$$
\mathbf p^i = \begin{bmatrix}
\sin(i \cdot \lambda_1) \\
\cos(i \cdot \lambda_1) \\
\sin(i \cdot \lambda_2) \\
\cos(i \cdot \lambda_2) \\
\vdots \\
\sin(i \cdot \lambda_{\frac{h}{2}}) \\
\cos(i \cdot \lambda_{\frac{h}{2}}) \\
\end{bmatrix} \qquad (1.4)
$$

上面的式子为什么要被写成这个样子？这其实是为了后续讨论的过程中帮我们忽略掉一些不必要的细节。

我们知道位置（position）$i$和位置$j$之间，相差的距离是$\mid i-j \mid$。那么我们定义的position embedding之间是否也有距离的概念？又是如何度量的？我们在position embedding层面上，定义的位置$i$跟位置$j$之间差异并不是简单的绝对值差或者欧几里得距离，而是**位置向量之间跟角度相关的差异**。

这种向量角度之间的差异又是如何度量的？我们通常会想到，一个向量可以通过乘以一个`旋转矩阵`变成另外一个向量。无论是Sinusoidal Position Embedding （绝对位置编码）还是后面介绍的Rotary position embedding，其设计原则中都离不开`旋转矩阵`这个概念。首先我们回顾一下这个概念。

## 1.2 旋转矩阵

什么是旋转矩阵？我们通过下面的一个例子给大家说明：

<img src="/assets/img/image-20240604231351392.png" alt="image-20240604231351392" style="zoom:25%;" />

假设我们有两个单位向量$\overrightarrow{OA}$以及$\overrightarrow{OB}$ ，他们和$x$轴之间的夹角分别为$\alpha$以及$\beta$，且$\beta - \alpha = \theta$，显然我们可以使用下面的等式（列向量，其中第一行表示$x$坐标值，第二行表示$y$坐标值）来表达$\overrightarrow{OA}$以及$\overrightarrow{OB}$：

$$ \overrightarrow{OA} =  \begin{bmatrix} \cos \alpha  \\ \sin \alpha \end{bmatrix}$$ 且 $$\overrightarrow{OB} =  \begin{bmatrix} \cos \beta  \\ \sin \beta \end{bmatrix} $$

我们要想把$\overrightarrow{OA}$变成$\overrightarrow{OB}$，所需要做的就是将$\overrightarrow{OA}$横坐标$cos \alpha$变成$\overrightarrow{OB}$的横坐标$cos \beta$；并且把$\overrightarrow{OA}$纵坐标$sin \alpha$变成$\overrightarrow{OB}$的纵坐标$sin \beta$；

我们回想一下上面的定义，$\alpha$和$\beta$之间差$\theta$，所以根据三角函数，我们有：

$cos \beta = cos (\alpha + \theta) = cos \alpha \cdot cos \theta  - sin \alpha \cdot sin \theta \qquad (1.2.1)$

如果我们把式子$(1.2.1)$使用矩阵乘法来写，可以表达为：$$ \cos \beta = \begin{bmatrix} \cos \theta & -\sin \theta \end{bmatrix}  \begin{bmatrix} \cos \alpha \\ \sin \alpha \end{bmatrix} \qquad (1.2.2) $$

同样地，对于$\overrightarrow{OB}$的横坐标$\sin \beta$，我们可以写作：$$ \sin \beta =  \sin(\alpha + \theta) = \sin\alpha \cdot \cos \theta + \cos \alpha \cdot \sin \theta = \begin{bmatrix} \sin\theta & \cos \theta \end{bmatrix} \begin{bmatrix} \cos \alpha \\ \sin \alpha \end{bmatrix} \qquad (1.2.3) $$

我们注意到$(1.2.2)$和$(1.2.3)$式的最右边实际上就是$\overrightarrow{OA}$，等号的左边则分别是$\overrightarrow{OB}$的横纵坐标坐标，所以如果使用矩阵相乘来同时表达$(1.2.2)$和$(1.2.3)$式，则我们有：

$$ \begin{bmatrix} \cos \beta \\ \sin \beta \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} \cos \alpha \\ \sin \alpha \end{bmatrix} $$

即：$$\overrightarrow{OB} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \overrightarrow{OA} \qquad (1.2.4)$$

即向量$\overrightarrow{OB}$可以通过一个角度为$\theta$的旋转矩阵乘上向量$\overrightarrow{OA}$来获得。

这个时候，我们将$$\begin{bmatrix} cos \theta & -sin \theta \\ sin \theta & cos \theta \end{bmatrix}$$ 称之从$\overrightarrow{OA}$到$\overrightarrow{OB}$的**旋转矩阵**；



## 1.3  不同position embedding之间的距离

 我们在前面提到了Sinusoidal Position Embedding在不同位置上的embedding：$\mathbf p^i$以及$\mathbf p^j$（假设$i < j$）之间的距离是通过两个向量之间角度差来定义的的，下面我们来具体看一下这一点在Sinusoidal Position Embedding上是如何实现的。

首先，我们定义$\mathbf p^i$在波长为$\lambda_m$ ($m \in \{1,...\frac{h}{2} \}$)的时候一个分量$$\mathbf p^i_{2m-1, 2m} = \begin{bmatrix} \mathbf p^i_{2m-1} \\ \mathbf p^i_{2m} \end{bmatrix}$$ 根据公式（1.4）的定义，实际上它们具有下面的取值（对于$\mathbf p^j$同理）：

$$
\begin{align*}
\mathbf p^i_{2m-1, 2m} &:= \begin{bmatrix}
\sin(i \cdot \lambda_m) \\
\cos(i \cdot \lambda_m)
\end{bmatrix} \\

\mathbf p^j_{2m-1, 2m} &:= \begin{bmatrix}
\sin(j \cdot \lambda_m) \\
\cos(j \cdot \lambda_m)
\end{bmatrix}
\end{align*} \qquad (1.3.1)
$$

根据1.2节中介绍的**旋转矩阵**的概念，从$\mathbf p^i_{2m-1,2m}$到$\mathbf p^j_{2m-1,2m}$，实际上应该是"大概"旋转了$(j-i)\lambda_m$ 这么大的弧度，这个具体的弧度是多大，我们通过下面完整的推导过程给出：

$$
\begin{align*}
\mathbf p^j_{2m-1,2m} &= \begin{bmatrix}
\sin(j \cdot \lambda_m) \\
\cos(j \cdot \lambda_m)
\end{bmatrix} \\

&= \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix} \begin{bmatrix}
\cos(j \cdot \lambda_m) \\
\sin(j \cdot \lambda_m)
\end{bmatrix} \qquad \# \text{为适配1.2节中的旋转矩阵颠换上下两行的位置} \\

&= \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix} \begin{bmatrix}
\cos((j-i)\lambda_m) & -\sin((j-i)\lambda_m) \\
\sin((j-i)\lambda_m) & \cos((j-i)\lambda_m)
\end{bmatrix} \begin{bmatrix}
\cos(i\cdot \lambda_m) \\
\sin(i \cdot \lambda_m)
\end{bmatrix} \qquad \# \text{对上一步中的第二个矩阵，套用1.2节中的公式(4)} \\

&= \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix} \begin{bmatrix}
\cos((j-i)\lambda_m) & -\sin((j-i)\lambda_m) \\
\sin((j-i)\lambda_m) & \cos((j-i)\lambda_m)
\end{bmatrix} \left ( 
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
\sin(i\cdot \lambda_m) \\
\cos(i \cdot \lambda_m)
\end{bmatrix} \right )  \qquad \# \text{上一步中最后一个矩阵拆分为两个矩阵} \\

&= \begin{bmatrix}
\sin((j-i)\lambda_m) & \cos((j-i)\lambda_m) \\
\cos((j-i)\lambda_m) & -\sin((j-i)\lambda_m) 
\end{bmatrix} \begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\begin{bmatrix}
\sin(i\cdot \lambda_m) \\
\cos(i \cdot \lambda_m)
\end{bmatrix} \qquad \# \text{将上一步中最左侧两个矩阵相乘} \\

&= \begin{bmatrix}
\cos((j-i)\lambda_m) &  \sin((j-i)\lambda_m) \\
-\sin((j-i)\lambda_m) & \cos((j-i)\lambda_m)
\end{bmatrix} \begin{bmatrix}
\sin(i\cdot \lambda_m) \\
\cos(i \cdot \lambda_m)
\end{bmatrix} \qquad \# \text{将上一步中最左侧两个矩阵相乘，实际是最左侧矩阵调换左右两列} \\

&= \begin{bmatrix}
\cos((j-i)\lambda_m) &  \sin((j-i)\lambda_m) \\
-\sin((j-i)\lambda_m) & \cos((j-i)\lambda_m)
\end{bmatrix} \cdot \mathbf p^i_{2m-1,2m} \qquad \# \text{将定义代入}  \\

&= \begin{bmatrix}
\cos((i-j)\lambda_m) &  -\sin((i-j)\lambda_m) \\
-\sin((i-j)\lambda_m) & \cos((i-j)\lambda_m)
\end{bmatrix} \cdot \mathbf p^i_{2m-1,2m} \qquad \# \text{将i-j调整为j-i,并使得和原矩阵相同} \\

&= \begin{bmatrix}
\cos(\theta') &  -\sin(\theta') \\
-\sin(\theta') & \cos(\theta')
\end{bmatrix}  \cdot \mathbf p^i_{2m-1,2m} \qquad (1.3.2)\quad\# \text{令} \theta'= (i-j)\lambda_m

\end{align*}
$$

所以，从式子（1.3.2）最后的结果可以看出来，$\mathbf p^j_{2m-1,2m}$实际上是$\mathbf p^i_{2m-1,2m}$旋转了$\theta' = (i-j)\lambda_m$ 弧度之后的结果。

在清楚了$\mathbf p^j_{2m-1,2m}$与$\mathbf p^i_{2m-1,2m}$之间的转换关系之后，我们应该如何表达$\mathbf p^j$与$\mathbf p^i$之间的转换关系呢？

假设我们令$k = j - i$，且根据公式（1.3.2），令$$\Phi^k_m = \begin{bmatrix} \cos(-k\lambda_m) & -\sin(-k\lambda_m) \\ -\sin(-k\lambda_m) & \cos(-k\lambda_m) \end{bmatrix}$$，即

$$
\mathbf p^{i+k}_{2m-1,2m} = \Phi^k_m \cdot \mathbf p^i_{2m-1,2m} \qquad (1.3.3)
$$

如果我们令：

$$
\mathbf T^{k} = \begin{bmatrix}
\Phi_1^k & \mathbf 0 & \cdots & \mathbf 0 \\
\mathbf 0 & \Phi_2^k & \cdots & \mathbf 0 \\
\mathbf 0 & \mathbf 0 &\ddots & \mathbf 0 \\
\mathbf 0 & \mathbf 0 & \cdots &\Phi^k_{\frac{h}{2}}
\end{bmatrix}  \qquad (1.3.4)
$$

且$\mathbf 0$表示$2 \times 2$全0矩阵，显然$\mathbf T^k \in \mathbb R^{h \times h}$

那么我们很容易验证：

$$
\mathbf p^j = \mathbf T^k  \mathbf p^i \qquad (1.3.5)
$$

这个结论看着相当的简单，它反应出了绝对位置编码下，$\mathbf p^i$以及$\mathbf p^j$之间存在一个仅仅与他们之间位置差异$k$相关变换$\mathbf T^k$。即单纯从位置编码的意义上，绝对位置编码也提供了很好的相对位置差的概念。但加入$\mathbf x^i$以及$\mathbf x^j$之后，这种数学上完美的相对位置差变得不再完整。后面我们从相对位置编码一节中可以看到这种不完整性。



## 1.4 Sinusoidal position embeding的实现

对于代码实现绝对位置编码感兴趣的同学，可以参考下面的实现。

```python
import numpy as np

def get_position_embs(seq_len:int, hidden_dim:int):
  """
  Args:
  	seq_len: sequence length, the returned position embeddings will have the same length.
  	hidden_dim: the size of hidden dimension of current model, usually the hidden_dim is equal to the word's embedding size.
  Returns:
  	A torch tensor, in shape (1, seq_len, hidden_dim)
  """
  def get_position_angle_vec(position, hidden_dim):
      return [position / np.power(10000, 2 * (hid_j // 2) / hidden_dim) for hid_j in range(hidden_dim)]

  # position_angle_vecs.shape = [seq_len, hidden_dim]
  position_angle_vecs = np.array([get_position_angle_vec(pos_i) for pos_i in range(seq_len)])

  # 分别计算奇偶索引位置对应的 sin 和 cos 值
  position_angle_vecs[:, 0::2] = np.sin(position_angle_vecs[:, 0::2])  # dim 2t
  position_angle_vecs[:, 1::2] = np.cos(position_angle_vecs[:, 1::2])  # dim 2t+1

  # positional_embeddings.shape = [1, seq_len, hidden_dim]
  positional_embeddings = torch.FloatTensor(position_angle_vecs).unsqueeze(0)
  return positional_embeddings

if __name__ == '__main__':
  pos_embs = get_position_embs(20, 512)
```



# 2. 相对位置编码

介绍完了绝对位置编码，下面我们再来看一下什么是相对位置编码？

首先，从公式（0.2）中，我们知道$\mathbf q_i$和$\mathbf k_j的取值如下$：

$$
\begin{align*}
\mathbf q_i &= \mathbf W_q(\mathbf x^i + \mathbf p^i) \\
\mathbf k_j &= \mathbf W_k(\mathbf x^j + \mathbf p^j) 
\end{align*} \qquad (2.1)
$$

那么依照公式（0.3），我们可以计算位置$i$的query对位置$j$上的key的权重的分子的部分如下：

$$
\begin{align*}
\mathbf q_i^\intercal \mathbf k_j &= (\mathbf x^i + \mathbf p^i)^\intercal \mathbf W_q^\intercal \mathbf W_k (\mathbf x^j + \mathbf p^j) \\
&= (\mathbf x^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \mathbf x^j  + (\mathbf p^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \mathbf x^j + (\mathbf x^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \mathbf p^j + (\mathbf p^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \mathbf p^j
\end{align*} \qquad (2.2)
$$

从公式（2.2）中，我们发现，如果有两个token：$t_1$和$t_2$，它们在一个序列中成对出现过两次，两次的出现位置分别为($m, m+20$)，($m+100, m + 120$) 

那么当$t_1$为query，$t_2$为key的时候，根据公式(2.2)，$\mathbf q_m^\top \mathbf k_{m+20}$跟$\mathbf q_{m+100}^\top \mathbf k_{m+120}$ 显然有很大的不同，虽然两组q,k之间的位置差异都是20个token。

所谓”相对位置编码“，其实就是基于现有的$\mathbf q_i^\intercal \mathbf k_j$的计算方式，做调整，从而将$\mathbf p^i$以及$\mathbf p^j$这种绝对位置编码从公式（2.2）中剔除。

具体的方法包括：

* 将公式（2.2）中的第二以及第四个求和项中的$\mathbf p^i$分别替换为两个可训练的参数$\mathbf u$以及$\mathbf v$ (其中$\mathbf u,\mathbf v$是跟位置无关的特征)
* 将公式（2.2）中的第三以及第四个求和项中的$\mathbf p^j$替换为一个sinusoid编码方法下的相对位置编码：$\mathbf{ \tilde{p}}_{i-j}$

那么，我们就可以得到如下的，只具有相对位置编码计算的$\mathbf q_i^{\intercal} \mathbf k_j$的新计算公式：

$$
\begin{align*}
\mathbf q_i^\intercal \mathbf k_j 
= (\mathbf x^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \mathbf x^j  + (\mathbf u)^\intercal \mathbf W_q^\intercal \widetilde{\mathbf W_k} \mathbf x^j + (\mathbf x^i)^\intercal \mathbf W_q^\intercal \mathbf W_k \tilde{\mathbf p}^{j-i} + (\mathbf v)^\intercal \mathbf W_q^\intercal \widetilde{\mathbf W_k} \tilde{\mathbf p}^{i-j}
\end{align*} \qquad (2.3)
$$


显然在公式（2.3）中，我们发现绝对位置编码消失了，只有相对位置编码$\tilde{\mathbf p}^{i-j}$，以及两个可训练的参数$\mathbf u,\mathbf v$。

还有更多的关于相对位置编码的思路，这里不再赘述，但核心思路都是在原有Sinusoidal position embedding的编码公式上做调整，使得相对位置编码$\tilde {\mathbf p}^{i-j}$成为attention计算中，唯一跟位置相关的特征。



# 3. 旋转位置编码



## 3.1 历史回顾



在介绍旋转位置编码之前，我们有必要再整体回顾一下，绝对位置编码（Sinusoidal Position Embedding，下文简称SPE）以及相对位置编码（Relative Position Embedding，下文简称RPE），是如何将**位置**信息编码到神经网络中的。

* 在SPE中，将一个序列中位置嵌入$\mathbf p^i$编码到神经网络中的方法是，将位置$i$上的词嵌入$\mathbf x^i$ 和位置嵌入$\mathbf p^i$，进行相加即$\mathbf x^i + \mathbf p^i$ （网络上有人说为什么是相加？而不是相乘，例如$\mathbf x^i \otimes \mathbf p^i$（其中$\otimes$表示Element-wise的乘法），网友们给出了各种解释，其实从**神经网络可拟合任何函数**的角度来说，只要**信息流动具有通畅性**，什么方法都行，当然了不同的方法肯定有不一样的训练难度以及效率不同。 其实行不行最后也都还是看“疗效”，即最终的结果）。 
* 在SPE中，所有的位置$i$都使用的绝对位置编码$\mathbf p^i$，但不同位置$i,j$ 之间的位置编码存在着以$k=i-j$的线性关系，即$\mathbf p^j = \mathbf T^k  \mathbf p^i$ (推导细节见上面的公式（13）)，且$\mathbf T^k$是某种形式的**旋转矩阵**，所以可以说$\mathbf p^i$跟$\mathbf p^j$的**相对位置**，实际上是某种形式的**相对角度**
* 在RPE中，绝对位置编码的概念消失了，取而代之的是位置$i,j$之间的相对位置编码：$\tilde{ \mathbf p}^{i-j}$。这里的$\tilde{ \mathbf p}^{i-j}$可使用Sinusoidal Position Embedding进行编码，也可以是一个可训练的模型参数（例如最长的编码长度为$L$，位置嵌入的维度数为$d$，则需要一个大小为$\mathbb R^{L \times d}$的可训练参数）

在这些工作的基础上，**旋转位置编码**被提出。在旋转位置编码中，序列中的每一个位置$i$，对应一个角度$\theta_i$；不同的位置$i,j$之间相对差，被转化为两个位置之间存在的角度差$\theta_i - \theta_j$；词嵌入$\mathbf x^i$中加入位置编码的方式，实际上是将向量$\mathbf x^i$沿着$\theta_i$的方向进行旋转。



## 3.2 旋转位置编码的定义以及简单证明

下面我们看一下其在原始论文中正式的定义：

旋转位置编码本质上是一种相对位置编码，即我们希望$\mathbf q_i^\intercal \mathbf k_j$的结果如同公式（2.3）中所示那样，只与$\mathbf x^i$，$\mathbf x^j$ 以及$i-j$相关，而与$\mathbf p^i$，$\mathbf p^j$无关。即假设存在下面的三个函数$f_q(\cdot), f_k(\cdot), g(\cdot)$是满足我们要求的的三个函数，即：

$$
\begin{align*}
\mathbf q_i &= f_q(\mathbf x^i, \mathbf p^i) \\
\mathbf k_j &= f_k(\mathbf x^j, \mathbf p^j) \\
\mathbf q_i^\intercal \mathbf k_j &= \langle \mathbf q_i, \mathbf k_j \rangle = g(\mathbf x^i, \mathbf x^j, i-j) \qquad \# \langle \mathbf a,\mathbf b \rangle 表示向量\mathbf a与\mathbf b之间的点积
\end{align*} \qquad   (3.1)
$$

在原始论文中作者给出了一组在位置嵌入的维度$h=2$的时候，可能的$f_q(\cdot)$，$f_k(\cdot)$，$g(\cdot)$的实现如下【在下面的实现中，作者使用了复数，其中使用了字母$i$表示虚数，为了防止符号混淆，从这里开始，我们将使用字母$m,n$替代本文中持续沿用的位置$i,j$】：

$$
\begin{align*}
f_q(\mathbf x^m, m) &= (\mathbf W_q \mathbf x^m) e^{im\theta} \\
f_k(\mathbf x^n, n) &= (\mathbf W_k \mathbf x^n) e^{in\theta} \\
g(\mathbf x^m, \mathbf x^n, m-n) &= Re[(\mathbf W_q \mathbf x^m)(\mathbf W_k \mathbf x^n)^*e^{i(m-n)\theta}]
\end{align*} \qquad (3.2)
$$

其中需要特别说明的是$\mathbf W_q \mathbf x^m$, $\mathbf W_k \mathbf x^n$在这里是两个独立的复数，而不是复数的欧拉表达形式$\mathcal z = r e^{\theta j} $中的“实部”. （例如，我们一般使用$\mathcal z = r e^{\theta j}$表示一个复数，其中$r$为复数的模，$\theta$为幅角）。

作为矩阵，$\mathbf W_q \mathbf x^m \in \mathbb R^{2 \times 1}$，所以$\mathbf W_q \mathbf x^m$的普通复数表达形式为：$\mathbf q_m^{(1)} + \mathbf q_m^{(2)}i$

对公式（3.2）的第一个式子，我们展开之后，可以得到：

$$
\begin{align*}
f_q(\mathbf x^m, m) &= (\mathbf W_q \mathbf x^m) e^{im\theta} \\
&= (\mathbf q_m^{(1)} + \mathbf q_m^{(2)}i)(\cos m\theta + i \sin m\theta) \\
&= (\mathbf q_m^{(1)} \cos m\theta - \mathbf q_m^{(2)} \sin m\theta) + i (\mathbf q_m^{(2)} \cos m\theta + \mathbf q_m^{(1)} \sin m\theta) \\
&:= \begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & cos m\theta
\end{bmatrix} 
\begin{bmatrix}
\mathbf q_m^{(1)} \\
\mathbf q_m^{(2)}
\end{bmatrix} \qquad \# 重新整理，得到相应的矩阵相乘形式; 矩阵的第一行表示复数的实部，第二行表示复数的虚部；


\end{align*} \qquad (3.3)
$$



同理：

$$
\begin{align*}
f_k(\mathbf x^n, n) &= (\mathbf W_k \mathbf x^n)e^{im\theta} \\

&= (\mathbf k_n^{(1)} + \mathbf k_n^{(2)}i)(\cos n\theta + i \sin n\theta) \\

&= (\mathbf k_n^{(1)} \cos n\theta - \mathbf k_n^{(2)} \sin n\theta) + i(\mathbf k_n^{(2)} \cos n\theta + \mathbf k_n^{(1)} \sin m\theta) \\

&:= \begin{bmatrix}
\cos n\theta & -\sin n\theta \\
\sin n\theta & cos n\theta
\end{bmatrix} 
\begin{bmatrix}
\mathbf k_n^{(1)} \\
\mathbf k_n^{(2)}
\end{bmatrix} \qquad \# 将上面的复数式子重新整理，得到相应的矩阵相乘形式（同上面3.3）
\end{align*} \qquad (3.4)
$$


这时候再来求$\langle f_q(\cdot), f_k(\cdot) \rangle$如下，

$$
\begin{align*}
\langle f_q(\mathbf x^m,m), f_k(\mathbf x^n, n) \rangle &= f_q(\mathbf x^m, m)^\intercal  f_k(\mathbf x^n, n) \\
&= \left(  \begin{bmatrix} 

\cos m\theta & -\sin m\theta \\
\sin m\theta & cos m\theta
\end{bmatrix} 
\begin{bmatrix}
\mathbf q_m^{(1)} \\
\mathbf q_m^{(2)}
\end{bmatrix}
\right)^\intercal 

\left ( 
\begin{bmatrix}
\cos n\theta & -\sin n\theta \\
\sin n\theta & cos n\theta
\end{bmatrix} 
\begin{bmatrix}
\mathbf k_n^{(1)} \\
\mathbf k_n^{(2)}
\end{bmatrix}
\right) \\

&=  [\mathbf q_m^{(1)}, \mathbf q_m^{(2)}] \begin{bmatrix}
\cos m\theta & \sin m\theta \\
-\sin m\theta & \cos m\theta
\end{bmatrix} 
\begin{bmatrix}
\cos n\theta & -\sin n\theta \\
\sin n\theta & cos n\theta
\end{bmatrix} 
\begin{bmatrix}
\mathbf k_n^{(1)} \\
\mathbf k_n^{(2)}
\end{bmatrix} \\

&= [\mathbf q_m^{(1)}, \mathbf q_m^{(2)}] \begin{bmatrix}
\cos((m-n)\theta) &   \sin((m-n)\theta) \\
-\sin((m-n)\theta) & \cos ((m-n)\theta)
\end{bmatrix} \begin{bmatrix}
\mathbf k_n^{(1)} \\
\mathbf k_n^{(2)}
\end{bmatrix} \\

\end{align*} \qquad (3.5)
$$

虽然，式子（3.5）已经是我们希望要的$g(\mathbf x^m, x^n, m-n)$的形式，至于式子（3.5）是否与$ Re[(\mathbf W_q \mathbf x^m)(\mathbf W_k \mathbf x^n)^*e^{i(m-n)\theta}]$等价，我们在后续的博文中继续证明这一部分的相等我们放在下一节进行介绍。

当把所有词嵌入的维度从2扩展到正常的词嵌入维度的时候，所面临的情况实际上跟1.3节完全相同，处理方式也完全相同。这里就不再赘述。



## 3.3 旋转位置编码的证明

首先，让我们回顾上一节，在3.2节中，我们给出一个满足式子（3.1）的要求的一组函数（3.2）

$$
\begin{align*}
f_q(\mathbf x^m, m) &= (\mathbf W_q \mathbf x^m) e^{im\theta} \\
f_k(\mathbf x^n, n) &= (\mathbf W_k \mathbf x^n) e^{in\theta} \\
g(\mathbf x^m, \mathbf x^n, m-n) &= Re[(\mathbf W_q \mathbf x^m)(\mathbf W_k \mathbf x^n)^*e^{i(m-n)\theta}]
\end{align*} \qquad (3.2)
$$


但并未严格证明$\langle f_q(\mathbf x^m, m), f_k(\mathbf x^n, n) \rangle = g(\mathbf x^m, \mathbf x^n, m-n) \quad (3.6)$

在正式证明之前，我们首先明确在式子（3.2）中，$(\mathbf W_q \mathbf x^m)$以及$(\mathbf W_k, \mathbf x^n)$都是复数，且$(\mathbf W \mathbf x^k)e^{ik\theta}$（$k=m或者n$）表示复数$(\mathbf W \mathbf x^k)$与复数$e^{ik\theta}$之间相乘。

如果我们将$(\mathbf W_q \mathbf x^m)$写成一个基于欧拉公式（Euler form）的形式$re^{i\theta}$的复数如下：

$$
\mathbf W_q \mathbf x^m = r_{qm} e^{i\theta_{mq}} \qquad (3.7)
$$

那么：

$$
\begin{align*}

f_q(\mathbf x^m, m) &= (\mathbf W_q \mathbf x^m) e^{im\theta} \\
&= r_{qm} e^{i \theta_{mq}} e^{im\theta} \\
&= r_{qm}(\cos \theta_{mq} + i\sin \theta_{mq})(\cos m\theta + i\sin m\theta) \\
&= r_{qm}((\cos \theta_{mq} \cos m\theta - \sin \theta_{mq} \sin m\theta) + i(\sin \theta_{mq} \cos m\theta +  \cos \theta_{mq} \sin m\theta)) \\
&= r_{qm}(\cos(\theta_{mq} + m\theta) + i \sin(\theta_{mq} + m\theta)) \\
&= r_{qm}e^{i(\theta_{mq} + m \theta)}

\end{align*} \qquad (3.8)
$$


同理，我们可以将$\mathbf W_k \mathbf x^n$写成欧拉公式形式的复数：

$$
\mathbf W_k \mathbf x^n = r_{kn} e^{i \theta_{kn}} \qquad (3.9)
$$


且跟式子（3.8）同理，我们可以得到$f_k(\mathbf x^n, n) = r_{kn} e^{i(\theta_{kn} + n\theta)} \quad (3.10)$

接着我们来回顾一下，**两个复数的点乘** 是什么？似乎很难定义，我们通过一个具体的例子来看一下。假设，有任意两个复数$a + bi$与$c+di$，我们通常希望这两个复数的**点乘**结果为$ac + bd$ ，（将复数的实部看做一个维度，虚部看做另外一个维度，则一个复数就是一个二维向量），那么我们如何通过$a+bi$与$c+di$的运算得到呢？ 如果我们使用$\bar{z}$表示复数$z$的共轭复数，那么$a+bi$的共轭复数 $\overline {a + bi} = a - bi$；且我们使用$Re(z)$表示复数$z$的实部；那么我们有：

$$
\begin{align*}
Re((a+bi)(\overline{c+di}))&= Re((ac+bd) + i(bc-ad)) \\
&= ac + bd \\
&= \langle a+bi, c+di \rangle
\end{align*} \qquad (3.11)
$$

我们实际上给出了一个定义，即：

$$
\langle a+bi, c+di \rangle = Re((a+bi)(\overline{c+di})) \qquad (3.11.1)
$$


我们再看一下共轭复数的欧拉形式表示是什么样的：

$$
\begin{align*}
\overline{ r e^{i\theta}} &= \overline{ r (\cos \theta + i \sin \theta)} \\
&= r(\cos \theta - i \sin \theta) \\
&= r(\cos (-\theta) + i \sin (-\theta)) \\
&= r e^{i(-\theta)}
\end{align*} \qquad (3.12)
$$

这时候，我们重新再来看我们要证明的等式（3.6）的左边如何展开：

$$
\begin{align*}
\langle f_q(\mathbf x^m,m), f_k(\mathbf x^n, n) \rangle &= Re( f_q(\mathbf x^m, m) \overline{f_k(\mathbf x^n, n)}) \qquad \#根据公式(3.11.1) \\
&= Re( r_{qm} e^{i (\theta_{mq} + m\theta)} \overline{ r_{kn} e^{i(\theta_{kn} + n\theta)} }) \qquad \#根据公式(3.8) \\
&= Re( r_{qm} e^{i (\theta_{mq} + m\theta)} r_{kn} e^{i (-\theta_{kn} - n\theta)} ) \qquad \#根据公式(3.12)\\
&= Re( r_{qm} r_{kn} e^{i[(\theta_{mq}-\theta_{kn}) + (m\theta - n\theta)]}) \\
&= Re( r_{qm} r_{kn} e^{i(\theta_{mq} - \theta_{kn})} e^{i(m-n)\theta}) \\
&= Re(r_{qm}e^{i\theta_{mq}}  r_{kn} e^{-i\theta_{kn}} e^{i(m-n)\theta}) \\
&= Re(r_{qm}e^{i\theta_{mq}}  \overline{r_{kn} e^{i\theta_{kn}}} e^{i(m-n)\theta}  ) \\
&= Re[(\mathbf W_q \mathbf x^m)  (\mathbf W_k\mathbf x^n)^* e^{i(m-n)\theta}] \qquad \#遵从原论文，使用z^*表示复数z的共轭复数 \\
&= g(\mathbf x^m, \mathbf x^n, m-n)
\end{align*}
$$

证明完毕。



## 3.4 旋转位置编码的实现



```python

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)
        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```

参考文献：

* https://www.cnblogs.com/invo/p/18243532 （复数的同构矩阵）
* https://zhuanlan.zhihu.com/p/642884818
* [Roformer](https://arxiv.org/pdf/2104.09864)
* [Sinusoidal Position Embeddings](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [T5 Bias](http://jmlr.org/papers/v21/20-074.html)


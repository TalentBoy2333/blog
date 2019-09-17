## 转发：https://zhuanlan.zhihu.com/p/63731947

现在，有一个分类问题：

- feature是2维的向量
- 目标类别有3种
- 一共有4个样本：$\left(x_{11}, x_{12}, y_{1}\right),\left(x_{21}, x_{22}, y_{2}\right),\left(x_{31}, x_{32}, y_{3}\right),\left(x_{41}, x_{42}, y_{4}\right)$

我们准备用一个只有一层的全连接神经网络来解决这个问题（使用多层神经网络推导太复杂，并且不利于理解，多层神经网络不是讲清此问题的关键）。

首先，我们需要使用one-hot来表示目标类别（为什么使用one-hot是另一个问题），所以，全连接神经网络的最后一层有三个数字。

哎呀！样本数太多了写起来很复杂，在计算loss时，只是简单地对多个样本产生的损失求均值，所以下面我们改一下问题：假设只有一个样本，这个样本的特征为 $\left(x_{1}, x_{2}\right)$，这个样本经过全连接神经网络之后输出为：$\left(y_{1}^{\prime}, y_{2}^{\prime}, y_{3}^{\prime}\right)$，真实值为$\left(y_{1}, y_{2}, y_{3}\right)$。
$$
\begin{aligned} y_{1}^{\prime}=\sigma\left(w_{11} \times x_{1}+w_{12} \times x_{2}+b_{1}\right) \\ y_{2}^{\prime}=\sigma\left(w_{21} \times x_{1}+w_{32} \times x_{2}+b_{2}\right) \\ y_{3}^{\prime}=\sigma\left(w_{31} \times x_{1}+w_{32} \times x_{2}+b_{3}\right) \\ \sigma(x)=\frac{1}{1+e^{-x}} \end{aligned}
$$
不妨设这个样本的类别为1，它的one-hot真实向量为(1,0,0)。

其实，在最后一层输出的时候，我们需要使用softmax把 $\left(y_{1}^{\prime}, y_{2}^{\prime}, y_{3}^{\prime}\right)$进行归一化。softmax的过程此处就省略了，就当$\left(y_{1}^{\prime}, y_{2}^{\prime}, y_{3}^{\prime}\right)$已经是softmax之后的结果了吧(因为softmax不是解释此问题的关键)。

下面看平方误差：
$$
\min \quad z=\frac{1}{2}\left[\left(y_{1}-y_{1}^{\prime}\right)^{2}+\left(y_{2}-y_{2}^{\prime}\right)^{2}+\left(y_{3}-y_{3}^{\prime}\right)^{2}\right]
$$
再看交叉熵误差：
$$
\min \quad z=y_{1} \log \frac{1}{y_{1}^{\prime}}+y_{2} \log \frac{1}{y_{2}^{\prime}}+y_{3} \log \frac{1}{y_{3}^{\prime}}
$$
其中，$y_{1}$表示真实值，$y_{1}^{\prime}$表示预测值。三部分是完全相同的，它们反向传播时效果是相似的。所以，我们只分析 $\min \quad z=\frac{1}{2}\left(y_{1}-y_{1}^{\prime}\right)^{2}$和$\min \quad z=y_{1} \log \left(y_{1}^{\prime}\right)$对权值的影响，和$y_{1}$有关的三个权值是$w_{11}, w_{12}, w_{13}$，别的权值不用看。我们只分析损失z对 $w_{11}$的影响。

对于平方误差：
$$
\begin{aligned} \frac{\partial z}{\partial w_{11}}=\left(y_{1}-y_{1}^{\prime}\right) \times \frac{\partial y_{1}^{\prime}}{\partial w_{11}} \\=\left(y_{1}-y_{1}^{\prime}\right) \times y_{1}^{\prime}\left(1-y_{1}^{\prime}\right) \times \frac{\partial\left(w_{11} x_{1}+w_{12} x_{2}\right)}{\partial w_{11}} \\=\left(y_{1}-y_{1}^{\prime}\right) \times y_{1}^{\prime}\left(1-y_{1}^{\prime}\right) \times x_{1} \end{aligned}
$$
我们想知道的是什么？我们想知道的是$\Delta w_{11}=f\left(y_{1}-y_{1}^{\prime}\right)$也就是$w_{11}$调整的幅度和绝对误差 $y_{1}-y_{1}^{\prime}$之间的关系。

记绝对误差$A=\left|y_{1}-y_{1}^{\prime}\right|$

因为我们使用了one-hot，所以$y_{1}$的真实值只能取0和1，而one-hot之后 $y_{1}^{\prime}$的值必然在0到1之间。

当$y_{1}=1$时，$A=y_{1}-y_{1}^{\prime}, \quad y_{1}^{\prime}=y_{1}-A$，代入$\frac{\partial z}{\partial w_{11}}$得到
$$
\frac{\partial z}{\partial w_{11}}=A \times\left(y_{1}-A\right) \times\left(1-\left(y_{1}-A\right)\right) \times x_{1}=A^{2}(1-A) \times x_{1}
$$
当$y_{1}=0$时，$A=y_{1}^{\prime}$，$\frac{\partial z}{\partial w_{11}}=-A^{2}(1-A) \times x_{1}$

此式中， $x_{1}$是常量，不必关心，我们只看 $A^{2}(1-A)$的形状

这个函数长啥样子？

```Python
import matplotlib.pyplot as plt
import numpy as np

A = np.linspace(0, 1, 100)
plt.plot(A, A ** 2 * (1 - A))
plt.xlabel("absolute error")
plt.ylabel("$\delta w_{11}$")
plt.title("$\delta w_{11}$=f(A)")
plt.show()
```

![1](/Users/zhangruitao/git/blog/为什么使用交叉熵而不是均方误差/1.jpg)

随着绝对误差的增大，权值需要调整的幅度先变大后变小，这就导致当绝对误差很大时，模型显得“自暴自弃”不肯学习

对于交叉熵误差：
$$
\begin{aligned} \frac{\partial z}{\partial w_{11}} &=y_{1} \times \frac{1}{y_{1}^{\prime}} \times \frac{\partial y_{1}^{\prime}}{w_{11}} \\=y_{1} \times \frac{1}{y_{1}^{\prime}} & \times y_{1}^{\prime} \times\left(1-y_{1}^{\prime}\right) \times x_{1} \\=& y_{1} \times\left(1-y_{1}^{\prime}\right) \times x_{1} \\=& y_{1} \times\left(1-\left(y_{1}-A\right)\right) \times x_{1} \\ \text {when } \quad y_{1}=1, \frac{\partial z}{\partial w_{11}} & \propto y_{1} \times A \times x_{1} \end{aligned}
$$
可以看到，使用交叉熵之后，绝对误差和需要调整的幅度成正比。

我们回过头来比较平方损失和交叉熵损失的区别，会发现：

平方损失的“罪魁祸首”是sigmoid函数求导之后变成$y_{1}^{\prime}\left(1-y_{1}^{\prime}\right) \propto(1-A) \times A^{2}$平白无故让曲线变得非常复杂，如果前面能够产生一个$\frac{1}{y_{1}^{\prime}}$把后面多余项“吃掉”多好

交叉熵的优势就是：它求导之后只提供了一个$\frac{1}{y_{1}^{\prime}}$去中和后面的导数。
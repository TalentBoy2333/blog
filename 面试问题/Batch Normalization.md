#### 个人理解 - 1
Batch Normalization的提出是为了解决模型在训练过程中每层网络不光要拟合分类任务, 还需要去拟合每次输入数据时不同的数据分布, 这导致了模型的不稳定. 至于其他的`加速收敛`, `保证梯度`, `缓解过拟合`等都是Batch Normalization改善了系统的结构合理性, 带来了一系列的性能改善. </br>

原文参考: </br>
While stochastic gradient is simple and effective, it requires careful tuning of the model hyper-parameters, specifically the learning rate used in optimization, as well as the initial values for the model parameters. The training is complicated by the fact that the inputs to each layer are affected by the parameters of all preceding layers —— so that small changes to the network parameters amplify as the network becomes deeper. </br>
翻译: </br>
虽然随机梯度是简单有效的, 但它需要仔细调整模型的超参数, 特别是优化中使用的学习速率以及模型参数的初始值. 训练的复杂性在于每层的输入受到前面所有层的参数的影响——因此当网络变得更深时, 网络参数的微小变化就会被放大. </br>

(也有人说, BN的核心思想不是为了防止梯度消失或者防止过拟合, 其核心是通过对系统参数搜索空间进行约束来增加系统鲁棒性, 这种约束压缩了搜索空间, 约束也改善了系统的结构合理性.) </br>


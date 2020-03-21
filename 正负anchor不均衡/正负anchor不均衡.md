### 问题描述：

检测任务。现在的检测方法如SSD和RCNN系列，都使用anchor机制。 训练时正负anchor的比例很悬殊。

除了类不平衡问题， 还有easy sample overwhelming的问题。



### 解决方案：

1. Hard Negative Mining，非online的mining/boosting方法， 以‘古老’的RCNN（2014）为代表， 但在CV里现在应该没有人使用了（吧？）。若感兴趣，推荐去看看OHEM论文里的related work部分。
2. Mini-batch Sampling，以Fast R-CNN（2015）和Faster R-CNN（2016）为代表。Fast RCNN在训练分类器， Faster R-CNN在训练RPN时，都会从N = 1或2张图片上随机选取mini_batch_size/2个RoI或anchor， 使用正负样本的比例为1：1。若正样本数量不足就用负样本填充。 使用这种方法的人应该也很少了。从这个方法开始， 包括后面列出的都是online的方法。
3. Online Hard Example Mining, OHEM（2016）。将所有sample根据当前loss排序，选出loss最大的N个，其余的抛弃。这个方法就只处理了easy sample的问题。
4. Online Hard Negative Mining, OHNM， SSD（2016）里使用的一个OHEM变种， 在Focal Loss里代号为OHEM 1：3。在计算loss时， 使用所有的positive anchor, 使用OHEM选择3倍于positive anchor的negative anchor。同时考虑了类间平衡与easy sample。
5. Class Balanced Loss。计算loss时，正负样本上的loss分别计算， 然后通过权重来平衡两者。暂时没找到是在哪提出来的，反正就这么被用起来了。它只考虑了类间平衡。
6. Focal Loss（2017）， 最近提出来的。不会像OHEM那样抛弃一部分样本， 而是和Class Balance一样考虑了每个样本， 不同的是难易样本上的loss权重是根据样本难度计算出来的。



————————————————
版权声明：本文为CSDN博主「Daniel2333」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_35653315/article/details/78327408
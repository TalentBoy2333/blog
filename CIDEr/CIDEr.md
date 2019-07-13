<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script> -->
## 文本挖掘预处理之TF-IDF
### 文本向量化特征的不足
在将文本分词并向量化后, 我们可以得到词汇表中每个词在各个文本中形成的词向量. </br>
我们将下面4个短文本做了词频统计: 
```Python
corpus=["I come to China to travel", 
        "This is a car polupar in China",          
        "I love tea and Apple ",   
        "The work is to write some papers in science"] 
```

不考虑停用词, 处理后得到的词向量如下: 
```Bash
[[0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 2 1 0 0]
 [0 0 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 0]
 [1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0]
 [0 0 0 0 0 1 1 0 1 0 1 1 0 1 0 1 0 1 1]]
```

如果我们直接将统计词频后的19维特征做为文本分类的输入, 会发现有一些问题. 比如第一个文本, 我们发现`"come"`, `"China"`和`"Travel"`各出现1次, 而`"to"`出现了两次. 似乎看起来这个文本与`"to"`这个特征更关系紧密. 但是实际上`"to"`是一个非常普遍的词, 几乎所有的文本都会用到, 因此虽然它的词频为2, 但是重要性却比词频为1的`"China"`和`"Travel"`要低的多. 如果我们的向量化特征仅仅用词频表示就无法反应这一点. 因此我们需要进一步的预处理来反应文本的这个特征. 而这个预处理就是TF-IDF. 

### TF-IDF
`TF-IDF`是`'Term Frequency - Inverse Document Frequency'`的缩写，即"词频-逆文本频率". 由两部分组成, `TF`和`IDF`. </br>

前面的`TF`也就是我们前面说到的词频, 我们之前做的向量化也就是做了文本中各个词的出现频率统计, 并作为文本特征, 这个很好理解. 关键是后面的这个`IDF`, 即"逆文本频率"如何理解. 在上一节中, 我们讲到几乎所有文本都会出现的`"to"`其词频虽然高, 但是重要性却应该比词频低的`"China"`和`"Travel"`要低. 我们的`IDF`就是来帮助我们来反应这个词的重要性的, 进而修正仅仅用词频表示的词特征值. </br>

概括来讲, `IDF`反应了一个词在所有文本中出现的频率, 如果一个词在很多的文本中出现, 那么它的IDF值应该低, 比如上文中的`"to"`. 而反过来如果一个词在比较少的文本中出现, 那么它的`IDF`值应该高. 比如一些专业的名词如`"Machine Learning"`. 这样的词`IDF`值应该高. 一个极端的情况. 如果一个词在所有的文本中都出现, 那么它的`IDF`值应该为`0`. </br>

上面是从定性上说明的IDF的作用, 那么如何对一个词的IDF进行定量分析呢? 这里直接给出一个词`x`的`IDF`的基本公式如下: </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/1.png" width = 25% height = 25% div align=center />

其中, `N`代表语料库中文本的总数, 而`N(x)`代表语料库中包含词`x`的文本总数。为什么`IDF`的基本公式应该是是上面这样的而不是像`N/N(x)`这样的形式呢? 这就涉及到信息论相关的一些知识了, 感兴趣的朋友建议阅读吴军博士的"数学之美"第11章. </br>

上面的`IDF`公式已经可以使用了, 但是在一些特殊的情况会有一些小问题, 比如某一个生僻词在语料库中没有, 这样我们的分母为`0`, `IDF`没有意义了. 所以常用的`IDF`我们需要做一些平滑, 使语料库中没有出现的词也可以得到一个合适的`IDF`值. 平滑的方法有很多种, 最常见的IDF平滑后的公式之一为: </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/2.png" width = 40% height = 40% div align=center />

有了`IDF`的定义, 我们就可以计算某一个词的`TF-IDF`值了: </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/3.png" width = 50% height = 50% div align=center />

其中`TF(x)`指词`x`在当前文本中的词频. </br>

## 余弦相似性
余弦相似度, 又称为余弦相似性, 是通过计算两个向量的夹角余弦值来评估他们的相似度. 余弦相似度将向量根据坐标值, 绘制到向量空间中, 如最常见的二维空间. </br>

两个向量间的余弦值可以通过使用欧几里得点积公式求出: </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/4.png" width = 25% height = 25% div align=center />

给定两个属性向量，A和B，其余弦相似性θ由点积和向量长度给出，如下所示: </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/5.png" width = 50% height = 50% div align=center />

这里的`$A_i$`, `$B_i$`分别代表向量A和B的各分量. </br>

给出的相似性范围从`-1`到`1`, `-1`意味着两个向量指向的方向正好截然相反, `1`表示它们的指向是完全相同的, `0`通常表示它们之间是独立的, 而在这之间的值则表示中间的相似性或相异性. </br>

对于文本匹配, 属性向量`A`和`B`通常是文档中的词频向量. 余弦相似性, 可以被看作是在比较过程中把文件长度正规化的方法. </br>

在信息检索的情况下, 由于一个词的频率(`TF-IDF`权)不能为负数, 所以这两个文档的余弦相似性范围从`0`到`1`. 并且, 两个词的频率向量之间的角度不能大于`90度`. </br>

## CIDEr算法
### 定义
待测数据集规模为N. </br>

候选集(Candidates) </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/6.png" width = 25% height = 25% div align=center />

参照集(References) </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/7.png" width = 25% height = 25% div align=center />

其中M表示参照集句子数量, i表示第i个图像. </br>

### TF-IDF
下面对候选集$c_i$, 计算其`n−gram`的`TF-IDF weight`. </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/8.png" width = 50% height = 50% align=center />

<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/9.png" width = 100% height = 100% align=left />
</br>

### CIDEr
n = [1, 2, 3, 4] 对应`n-gram`的n, 如`1-gram`, `2-gram`, `3-gram`, `4-gram`. </br>
<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/10.png" width = 50% height = 50% align=center />

<img src="https://github.com/TalentBoy2333/blog/blob/master/CIDEr/images/11.png" width = 50% height = 50% align=left />
</br>

## 参考文献
[1]https://www.cnblogs.com/pinard/p/6693230.html</br>
[2]https://baike.baidu.com/item/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E5%BA%A6/17509249?fr=aladdin</br>
[3]https://blog.csdn.net/wl1710582732/article/details/84202254</br>
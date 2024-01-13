预先的一些知识：
这个世界上的词和字的总数是有限的。
在实际的使用过程中，这些字词的组合也是有限的。
现实生活中，我们说过的话的总数也是有限的。

如果一门语言能够用它所包含的所有可能的话代替的话（语言即所有可能的句子的集合），那么正门语言也能用它所包含的所有字词和字词的组合关系表示。
如果把字词视为节点，将关系视为有向边，那么一门语言就是一个巨大的有向图。
一句话就是其中的一条有限的路径。
如果我们将每一个字词都用空间中的一个点表示，那么我们就可以将这个有向图画出来。那么这个有向图的包络就具有一个形状 —— 分布（语言空间）。
如果空间中的点之间的距离也反映了字词之间的某种语义关系的话，那么这个图还是语义图。
那么翻译其实就是将一个空间中的一条路径映射为另一个空间中的另一条路径。也就是将空间中的一个语义分布转换为另一种语义分布。这种转换肯定不是线性的。并且为了实现非线性映射，模型的维度肯定要比语义空间本身的维度更高，这样才能实现较好的非线性映射。

既然是空间分布的转换，那么其实就还是对空间本身的操作 —— 扭曲、分割、shearing、升维、降维、投影等。

另外，聚焦到一个字词，它前面可能有很多条路径，可能有很多个前置节点，然后它后面也可能跟了很多后置路径和后置节点。这个字词本身的含义可能有多重。具体含义还得由上下文决定。

而我们的模型其实就是需要对这种语言空间、路径以及多重语义进行建模。然后不仅仅是对一们语言进行建模，还需要对目标语言进行建模。不仅仅需要对两门语言进行建模，还需要对两门语言的模型进行对齐。这样才能实现翻译功能。


encoder 和 decoder的思路是，encoder从一种形式中提取出有用的表征信息，然后decoder则根据表征信息重建出新的形式。
人们想的是，我不知道需要提取那些表征信息，但是我给模型一些空间去存储这些信息，然后通过训练让模型自己去找。
但是从空间变换的角度来看，其实所谓的表征信息，就是一个中间过渡的空间分布罢了。这种过渡空间分布有点人为结构化原始空间分布的含义。因为人为设定了中间的空间大小和维度。

# 残差
突然意识到，残差的一个好处是，使得多层能在同一个向量中的不同位置写上自己的信息。就比如，一个512长度的向量。在残差过程中，可能第一块将有效信息写到向量的前十个元素里。第二层的残差块将有效信息写入到11-20这一个区间。以此类推。
# Dropout
Dropout的regularization的作用一目了然。因为将某些神经元的输出设置为了0，那么就相当于删掉了这些神经元。那么其实就相当于在拟合曲线的时候，删掉了某些点，那么在这个区间内，曲线就倾向于平滑，因为缺失了一个拉动曲线弯曲的点。
从信息传递的角度来看，则是因为屏蔽了部分神经元，那么其实就是在强制使得剩下的神经元以及对应的通路对某一类信息有更强烈的反应。也就是实现通路的分化。

参考项目：
* https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master
https://github.com/hyunwoongko/transformer 看图


# abstract 
1. 用attention机制替换掉了卷积和循环机制。
2. 并行性更高，因此运算更快。
3. 在机器翻译上的效果更好。
4. 泛化能力强。

# model architecture
总体来看，还是分为encoder和decoder两部分。encoder的输出是representation，这个序列的长度和输入序列的长度一致。
decoder则是一步一步运行的。每一步生成一个单词。每次的输入是encoder的输出和decoder自己之前的所有输出。这是一个auto-regressive 过程。
## Encoder and Decoder Stacks
### Encoder
It has 6 layers.
Each layer has two sub-layers: multi-head self-attention mechanism and position-wise fully connected feed-forward network.
Each layer also has two layer normalizations and two residual connections.

Encoder 的输入和输出，以及中间layer和sub-layer的输入和输出的维度都是 d = 512。

这里的两个sub-layers将在后面详细介绍。
输入值将在后面介绍。

### Decoder
N = 6 identical layers
three sub-layers
Using masking to prevent positions from attending to subsequent positions
输入值将在后面介绍。

在产生输出的过程中，首先输入的是<BOS>表示句子开始。然后decoder的最终的softmax层能预测出下一个单词。得到下一个单词之后就拿回到decoder的输入那里。继续预测下一个词。
关于最终的softmax输出概率的问题。有一种说法是，输出向量长度等于vocabulary的长度，然后向量元素的值就是每个词汇的概率。所以他通过这种方式预测下一个词也是可以的。但是我想的是，能不能像 diffusion 模型一样，输入是随机值，然后输出是对这个随机值的降噪结果。比如当前需要预测第一个单词了。那么首先给模型一个随机向量作为输入。然后在后续的自注意力阶段，这个随机向量在encoder给出的语义环境下不断收敛到第一个单词的embedding上。然后直接将这个embedding作为下一次的输入，同时再增加一个新的随机向量。如此往复循环，知道生成<EOS>的embedding。
最后再将这些embedding转化为单词输出？看看有没有人这样做？以及看看这样做的价值在哪里？
`如果是这样做的话，那么其实可以直接将带噪声的原始数据直接输入给decoder，然后就能直接一次性得到完整的输出。而不用递归了。这似乎能写篇论文。因为它能显著加快Decoder的生成速度。`



### Attention (multi-head sub-layer)
他说注意力机制的基本原理是，输入query、key和value，然后计算key和query之间的compatibility，然后将compatibility的值作为values的权重。将values的weighted sum作为注意力层的输出。
但是我这里不明白为什么不叫similarity，而要叫 compatibility？
`我想做的可扩展的能一直在线学习的模型是不是就可以照着这种query、key和value的方式来做呢？`

#### Scaled Dot-Product Attention
The input consists of queries and keys of dimension dk, and values of dimension dv.
query、value和key这三个元素都不是只有一个，三者都是一个序列。
注意，query和key两个向量的dot product是一个scalar。每一个query都和整个keys进行dot product。那么就得到一个scalar序列。这个 scalar序列在经过 softmax 之后刚好作为 values 序列的权重。
在做了weighted sum之后得到的还是一个向量。
然后对每个query 都做同样的操作，最终就能得到一系列的向量了——一个同输入一样长度的向量序列。

也就是说 Scaled Dot-Product Attention 这部分的三个输入分别是：query（一个向量）、keys（一系列向量）和values（一系列向量）

单纯从结构上和信息的角度来看，点积算的是向量投影，也就是query在keys上的投影，投影越大表示越正相关，0表示无关，负数表示负相关。那么，和query相关性越大的key所对应的value就应该得到越大的权重。这就是信息检索。最终的加权和就是对最终语义信息的近似。

从句子的角度看，query是一个单词的表达，那么，这个单词的含义是由上下文确定的，而上下文正式keys，含义则是key对query的含义的定义则为values。
所以这个attention过程就是在从语义的角度确定单词（query）的含义。
如果对句子中的每个单词都进行这种attention计算，那么就能得到对每个单词的含义的表达。那么这个结果就可以给decoder了。

#### Multi-Head Attention
在实验过程中，他们发现，如果将q、k和v先分别做线性映射，然后再做attention，再将得到的weighted sums连起来，做个线性映射，最后得到的效果会更好。
这里其实也可以理解。因为如果只有一个attention head的话，其实就只有一个信息通道。但是我们知道每一个单词的含义都是多重的。但是一个单词的embedding却只有一个。那么怎么展开这种多重含义呢？那就是上面说的做个线性映射，然后构建multi-head attention，从而对这种多重语义进行建模。而且，有时候一句话在不同的场景中其实有不同的含义的。所以这种不同还是需要多个attention通道来建模的。
最后的线性映射（线性层）其实就是个选择层。因为multi-head attention实际上是给出的多种可能性的结果，最终选择哪一个可能性呢？这就是线性层的工作了。其实无论是在任何一个NN中，网络的结构就构成了信息传递的通道。但是实际上在一个任务中，并不会完全激活每一个神经元和每一个信息通路。整个网络结构其实是具有冗余性的。线性层或者全连接层就是起到一个选择的作用（`滤波响应`）。

卷积层起到一个信息提取和组合的功能，全连接层起到信息组合和选择的功能。
在这里attention起到确定语义含义的功能，也就是起到信息提取的功能。
论文中也说了类似的话：
Multi-head attention allows the model to jointly attend to information from different `representation subspaces` at different positions. With a single attention head, averaging `inhibits` this.

注意在论文中，在计算representation subspaces的时候，对每个向量进行了降维，也就是对数据进行了压缩，也就是只选择了原始数据中的部分数据，这算不算是从语义集合中筛选出单个的语义呢？

注意论文中连接各个head的结果的时候，实际上是各个head的结果连接成一个向量，这个向量的维度是1 x d_model。在有些人的代码实现中，他们并没有严格设置Wq、Wk和Wv的维度，而是直接计算，然后将结果拆分成多个head。这样做在理论上能兼容论文中的做法。但是会增加额外的参数量。
所以对于一个query，最终multi-head的输出仍然是一个1 x d_model 的向量。

#### Applications of Attention in our Model
1. 在decoder的attention中，queries 来自decoder的上一层或者上次的输出。attention的keys和values则是来自encoder的输出。注意keys和values是一样的。
2. 在encoder这边，每一层的输入都是上一层的输出，此时的queries、keys和values都是一样的。区别在于之后的线性映射和attention阶段。多增加几层的原因可以理解为一层的映射能力不够。或者说一层的非线性空间扭曲能力不够，所以需要多增加几层。
3. masking的方法就是将需要被遮盖的部分的值设置为一个负很多的数，这样经过softmax之后其对应的之基本就为零了。

我这里还有的疑问是怎么产生第一个输出值的？以及怎么实现每次多输出一个值的？
Decoder的最后一层全连接层和Softmax 层能给出下一个词的概率分布，然后对下一个词进行采样。

### Position-wise Feed-Forward Networks (sub-layer)
这里是一个两层的全连接神经网络，两层中间包含了一个relu激活函数，但是第二层没有。
全连接层的输入和输出都是512维的。但是隐藏层的维度是2048。这种两头窄，中间宽的就是典型的高维非线性映射曲线。也就是在拟合从输出到输出的一个高精度的曲线。但是在这里起到的不是选择的作用，而仅仅是进行映射。作者说这里也可以理解为一次卷积。
一个position其实就是一个词。
这里的position-wise是指一个词对应的representation向量是FFN的输入，然后一个句子被视为一个batch。因为源码中并没有给出concatenate的操作。并且FFN的输出维度是512.


### Embeddings and Softmax
首先embedding是通过已经训练好的模型做的。并且encoder和decoder使用的embedding模型是同一个。
在得到embedding之后，还要对结果乘以一个权重（据说是更好的与position encoding 匹配）。
另外就是在decoder的输出之上，还增加了一个全连接层和softmax层。这是为了计算下一个词的概率。并且这里也是用了

### Positional Encoding
在有些情况下，单词的先后顺序决定了不同的含义。因此位置信息很重要。
这里作者说，他们做了实验，他们的position embedding算法和别人的模型效果基本一致。
但是他们的算法能够适应任意长度的输入，于是就采用了他们的算法。（但是他们怎么想到的？）
算法有两个参数，一个是向量维度i，另一个是单词position pos。（看图）
这有两个效果，第一每个pos的编码结果都不一样。第二，能够编码很长的sequence。


## Why Self-Attention
这一段看得不是很懂。明天继续看一遍。

作者提到他们提出这种注意力机制的三个原因：第一，层内计算复杂度；第二，可并行的计算数量；第三，记忆力长度（相关性）。
输入和输出中任意两个position之间的computation path越短，他们之间的相关性就学得越好。（也就是输出序列中的某个position和输入序列中的某个position之间的路径越短，学习的效果越好）

但是这里他们也说了存在不足。他们的一个假设是句子长度n往往比单词embedding的维度小很多。但是对于长文本，就只能限制一个上下文窗口。

最后还说，他们可视化了注意力分布，发现，不同的head能够完成不同的任务。并且很多heads都表现出和句法或语义相关的行为。这其实也验证了我们上面的猜想。你需要给模型不同的通道让它去对句子中不同的信息进行建模。


# words
`transduction` trænzˈdʌkʃən
[N]the transfer by a bacteriophage of genetic material from one bacterium to another 转导

`dispense with` 免除 ; 省掉 ; 无需 ; 省略

`Constituency Parsing` is the process of analyzing the sentences by breaking down it into sub-phrases also known as constituents. 

attend to / əˈtend tu / 关注 或 处理 某事物或某人。

counteract ˌkaʊntərˈækt (counteracting, counteracted, counteracts)
[V-T]To `counteract` something means to reduce its effect by doing something that produces an opposite effect. 对…起反作用; 抵消

inhibit ɪnˈhɪbɪt (inhibiting, inhibited, inhibits)
[V-T]If something inhibits an event or process, it prevents it or slows it down. 阻碍; 抑制

extrapolate ɪkˈstræpəˌleɪt (extrapolating, extrapolated, extrapolates)
[V-I]If you extrapolate from known facts, you use them as a basis for general statements about a situation or about what is likely to happen in the future. 推断

sinusoid 英/ ˈsaɪnəˌsɔɪd /美/ ˈsaɪnəsɔɪd / n.正弦曲线

desiderata 英/ dɪˌzɪdəˈreɪtə /美/ dɪˌzɪdəˈretəm /简明柯林斯n.（拉丁）迫切需要得到之物（desideratum 的复数）

contiguous / kənˈtɪɡjuəs /美/ kənˈtɪɡjuəs /简明柯林斯adj.连续的；邻近的；接触的
Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels,
在连续内核的情况下，这样做需要一堆 O(n/k) 卷积层，

dilated 英/ daɪˈleɪtɪd /美/ daɪˈleɪtɪd /简明柯林斯adj.扩大的；膨胀的；加宽的
or O(log k (n)) in the case of dilated convolutions
这样做需要在连续内核的情况下需要一堆 O(n/k) 卷积层，或者在扩张卷积的情况下需要 O(log k (n))

syntactic 英 / sɪnˈtæktɪk /美/ sɪnˈtæktɪk /简明柯林斯adj.句法的；语法的；依据造句法的
Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the `syntactic` and semantic structure of the sentences.
个体注意力头不仅清楚地学习执行不同的任务，而且许多注意力头似乎表现出与句子的句法和语义结构相关的行为。





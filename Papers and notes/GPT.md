# GPT-1
GPT-1 模型直接使用的是Transformer的decoder，当然他这里没有来自于encoder的启发信息了。所有信息都来自于decoder接收到的输入。
并且由于decoder逐个产生新词的特性，它是一个language model，对应的使用相应的损失函数。
然后，这篇文章中使用到的模型是12层，768的隐藏层宽度。
这篇文章的另一个创新点是，将多个自然语言处理的任务整合到一个统一的模板中。并且适配成文字接龙任务。

GPT-1的思路也还是先在没有标签的数据集上进行self-supervised的学习，或者叫semi-supervised learning。学好之后呢，再在子任务上进行监督学习微调。这时候往往会新增加一个全连接层，从而适配新的任务。

# GPT-2
这篇论文对于GPT-1而言就单纯是增大了模型。
然后这篇文章的重点在探索模型的zero-shot的能力。另一个重点在于更大的数据集的构建。
并且在这篇文章中的一个突破点是，他们取消了start、delimiter和extract等特殊字符，而是尝试直接将这些字符替换为自然语言的表达，如：翻译、回答等指令。这样模型就能直接学到这些指令的含义，从而为后来的in-context-learning 埋下伏笔。 


# GPT-3
这里的模型是基于GPT-2的模型，引入了Sparse Transformer的内容。
然后强调meta learning和in-context learning。
强调，在interaction 中的few-shot learning。






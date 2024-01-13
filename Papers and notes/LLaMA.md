关注点在模型


# 模型
论文中只说他们的模型是Based on Transformer architecture。但是并没有说具体是encoder还是decoder，那么应该就是两者都用了。
Like other large language models, LLaMA works by taking a sequence of words as an input and predicts a next word to recursively generate text. 

LLaMA 也是基于Transformer的，然后做了三个方面的改进（实际上是从别人的研究中借鉴过来，进行了一个组合优化）：
1. To improve the training stability, we normalize the input of each transformer `sub-layer`, instead of normalizing the output. We use the `RMSNorm normalizing` function.
2. We replace the ReLU non-linearity by the `SwiGLU` activation function, introduced by Shazeer (2020) to improve the performance.
3. We remove the absolute positional embeddings, and instead, add `rotary positional embeddings` (RoPE), introduced by Su et al. (2021), `at each layer` of the network.





# 重要性
LlaMA的主要优点：1 完全开源；2 性能超过GPT-3，Chinchilla-70B 和 PaLM-540B。但是参数量却小得多。
开源，小体量，高性能。





# 单词
resort rɪˈzɔːt (resorting, resorted, resorts)
[V-I]If you resort to a course of action that you do not really approve of, you adopt it because you cannot see any other way of achieving what you want. 不得不求助

proprietary prəˈpraɪɪtərɪ
[ADJ]Proprietary substances or products are sold under a brand name. 品牌专卖的，私有的
...some proprietary brands of dog food.…一些专卖狗粮的品牌。

without resorting to proprietary and inaccessible datasets.
无需诉诸专有且无法访问的数据集。

textual ˈtɛkstjʊəl
[ADJ]Textual means relating to written texts, especially literary texts. 文本的

rotary ˈrəʊtərɪ
1.[ADJ]Rotary means turning or able to turn around a fixed point. 旋转的


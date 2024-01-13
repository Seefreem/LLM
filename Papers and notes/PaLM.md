
https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html
https://blog.google/technology/ai/google-palm-2-ai-large-language-model/
https://ai.google/discover/palm2/


# PaLM
PaLM的模型采用了Transformer的decoder部分，然后做了挺多的修改的。
模型部分值得注意的几点：
1 取消了全连接层的bias
2 使用的vocabulary是能完全重建原句子的。它能无损保存空格，这对代码而言很重要。
3 使用了Multi-Query Attention
4 使用了SwiGLU Activation激活函数
5 将串联的全连接层改成了并联的全连接层

Multi-Query Attention – The standard Transformer formulation uses k attention heads, where the
input vector for each timestep is linearly projected into “query”, “key”, and “value” tensors of shape
[k, h], where h is the attention head size. Here, the key/value projections are shared for each head, i.e.
“key” and “value” are projected to [1, h], but “query” is still projected to shape [k, h]. 
正如上文所说。所有的头共享一个key，也共享一个value。这其实很有意思。这居然不影响结果。也就是说其中有可研究的东西。


注意一下model card。似乎还挺重要的。


# PaLM2
PaLM2 是在PaLM的基础之上进行了下面三个部分的改进：
1. Use of compute-optimal scaling: The basic idea of compute-optimal scaling is to scale the model size and the training dataset size in proportion to each other. This new technique makes PaLM 2 smaller than PaLM, but more efficient with overall better performance, including faster inference, fewer parameters to serve, and a lower serving cost.
2. Improved dataset mixture: Previous LLMs, like PaLM, used pre-training datasets that were mostly English-only text. PaLM 2 improves on its corpus with a more multilingual and diverse pre-training mixture, which includes hundreds of human and programming languages, mathematical equations, scientific papers, and web pages.
3. Updated model architecture and objective: PaLM 2 has an improved architecture. PaLM 2 and its latest version were trained on a variety of different tasks, all of which helps PaLM 2 learn different aspects of language.

但是只有一片技术报告，并没有对应的论文。






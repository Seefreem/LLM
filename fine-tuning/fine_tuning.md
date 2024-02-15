参考文章：
https://www.unite.ai/zh-CN/understanding-llm-fine-tuning-tailoring-large-language-models-to-your-unique-requirements/
https://zhuanlan.zhihu.com/p/620885226




关于微调的一些事实：
微调时需要的数据并不一定很多。
微调时训练的epoch数从一般只有几。
单任务微调可以通过相对较小的一组示例（范围从 500 到 1000 个）实现显着的性能增强。
微调只能提升模型的领域表现，并且减少出现错误的概率，但是并不能完全解决大模型本身的一些问题，比如幻觉，或者生成虚假的信息、有害的信息和带偏见的信息。
微调的两大类方法：RLHF和PEFT(Parameter-Efficient Fine-Tuning)

PEFT:
Additive Method: 
    Adapters: 也就是在模型顶部增加输出层，然后只训练输出层。
    Soft Prompts: 这个不是很懂。
    Other Additive Approaches: 没研究过。
Selective Method:
    这个和BERT的微调类似。也就是选择性地微调模型参数。比如微调某些层，微调bias等等。
Reparametrization-based Method:
    LoRA: 这个则是在每一层增加一个并行的SVD层，SVD对输入进行SVD分解，然后压缩，再还原到输入的维度，将SVD层的输出和原来模型层的输出相加。只训练新增加的SVD层。
    QLoRA: 量化版本的LoRA，但是还有创新。

但是LoRA训练的是什么内容呢？
你能自己训练一次不？


# 关于微调本身
Ref: https://www.bilibili.com/video/BV1Dm4y157Dc?p=1&vd_source=d6e01f1e452c2bebccda71a39bb0b20b

微调是指任何一种通过修改模型参数来改变模型行为的方法。修改模型行为是指，可以修改模型输出的方式，修改ChatBot 的聊天语气，以及让模型学习新的知识，从而避免输出错误的答案。这些都是微调。主要分为两大类，一类是改变模型的输出方式。另一种是让模型学习新的知识。当然也可以是两者的结合。

load 一个在线数据集的时候，Streaming是指一次只拿一部分数据，然后不断地拿。而不是一下口气下载所有的数据，然后再开始训练。
在准备QA类型的fine-tuning的时候，一般是把问题和答案分为两列，组成一个表格。这被称之为结构化的数据。我们可以直接将问题和答案连起来，组成一个字符串，然后输入给LLM。但是这通常不利于模型理解指令。
于是我们用到prompt template去增加一些提示词，比如：### Question:； ### Answer:。等等。
但是在存储数据的时候，我们是将加了prompt template 和 Question的字符串作为前半部分，将Answer的内容作为后半部分。因为这样方便将数据分为training set和 test set。

有个比较有意思的事情是，有些人用现有的ChatGPT来准备自己的训练集。比如给一句话，让ChatGPT做转述，做修改。从而实现同一个意思但是不同的表达。也可以用ChatGPT将一段不结构化的文本转化为结构化的文本。这样确实能够节省不少时间。或者就是将一段写的不是很好的话修改为表达更加优美的话。
而且这个过程其实可以重复。一开始拿到的模型可能不是很好。但是能够通过prompting让模型暂时变得比较符合我们的要求。然后在这种情况下生成一些数据。利用这些数据对模型进行微调。然后重复这个过程。

在微调的时候，其实总是可以将ChatGPT的输出作为你的参考(baseline)。

在tokenizing 的时候，有两个参数可以设置。一个是padding，因为一个batch中的句子长度应该一样。或者说输入给模型的文本长度是固定的，所以不足的部分就需要填充。反过来，超过的部分就需要truncate。


在evaluation的时候，其实没有很完美的指标。所以通常的做法是在多个任务上或者在一个任务上使用多个指标去评估。
多指标已经成为基本的操作了。
另外一个可以做的事情是，我们可以对预训练的模型进行error evaluation，这样我们就更清楚模型在哪些方面做得好，在哪些方面做得不好。然后我们可以针对性地进行微调。同样，我们在微调自己的模型之后也可以做error evaluation。

`LoRA的魅力不仅仅在于减少了训练的参数量。还在于它就像是一个插件一样，你可以在不同的数据集上微调出多个模型。然后分别保存它们的参数。在推理的时候，就根据任务类型动态组装微调后的模型。这其实就是他们说的插件啦。`

微调的实用指南：
1. Figure out your task.
2. Collect data related to the task's input/output.
3. Generate data if you don't have enough data.
4. Fine-tune a small model (e.g. 400m~1B) to see what performance it will get.
5. Vary the amount of data you give the model to understand how much data actually influences where the model is going.
6. Evaluate your LLM to know what's going well vs. not.
7. Collect more data to improve.
8. Increase task complexity.
9. Increase model size for performance.







# 词汇
SVD dissects a matrix into three distinct matrices, one of which is a diagonal matrix housing singular values. These singular values are pivotal as they gauge the significance of different dimensions in the matrices, with larger values indicating higher importance and smaller ones denoting lesser significance.

This is achieved by performing quantization on the already quantized constants, a strategy that averts typical gradient checkpointing memory spikes through the utilization of paged optimizers and unified memory management.

The 65B and 33B versions of Guanaco, fine-tuned utilizing a modified version of the OASST1 dataset, emerge as formidable contenders to renowned models like ChatGPT and even GPT-4.























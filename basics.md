- input_ids
token 在变为embedding之前还有一个id。整个过程是 tokens -> ids -> embeddings.
一般tokenizer 就是实现 ‘tokens -> ids’ 这个过程。
embedder 的作用是 实现 ‘ids -> embeddings’过程。

inputs_embeds = self.embed_tokens(input_ids)，其中embeds的shape是torch.Size([1, 8, 4096])。 

Ref： https://ai.oldpan.me/t/topic/169    https://www.cnblogs.com/marsggbo/p/17871464.html

# Facts about LLMs
1. Researches may train models for different tasks separately but using the same architecture or backbone. Then they would compare their evaluation results to other people's works. E.g. BART.

2. There are many downstream language tasks, generation is just one of them. Not all LLMs are trained for generating. Usually, Decoder-only architectures are used for generation, Encoder-included architectures are trained for classification tasks.

3. Many downstream tasks are defined as a classification task.

4. BERT-like architectures and GPT-like architectures are usually trained on different datasets, and their abilities are usually different.

5. Usually we can find source code from paper, github, huggingface, python-third-party library, etc.

6. I found the short introductions of terms about LLMs from the source file. 

7. input_ids `(batch_size, sequence_length)`
attention_mask `(batch_size, sequence_length)`
head_mask `(encoder_layers, encoder_attention_heads)`
inputs_embeds `(batch_size, sequence_length, hidden_size)`

8. 在Generate的时候，有些算法是采用了window attention，也就是每次只将新生成的token输入到 Decoder，然后计算其KV，并且预测下一个token。而当前的token的KV则被保存起来。以此循环，直到达到终止条件。这里确实存在一个循环。Decoder的任务其实就是预测下一个token。仅仅是这样而已。在产生下一个token的时候也存在一些算法，比如 BEAM_SEARCH, GREEDY_SEARCH, SAMPLE, BEAM_SAMPLE 等等。 

9. How to train a LLM?

10. How to evaluate a LLM?

11. How to use a LLM to generate sentence?
    1. Setting up the model
    2. loop:
        check stopping criteria;
        predict the probabilities of next token;
        sample next token;
        [save current KVs;]
        concatenate the new generated token as the end of generated sequence;
    
12. For Encoder-Decoder model, the instructions go into Encoder, and the Decoder only has to generate the answer. The output of the last layer of Encoder is the reference of Decoder. 
Then, the question is what will happen if I input documents for both Encoder and Decoder? 

13. For Decoder-only model, the instructions go into Decoder, as it has to generate reference for itself.

14. LLMs do not know when to stop generating. E.g. 'The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs. High winds are' (这里停止生成的原因是设置了最大的生成长度)

15. logits_processor() 函数是对模型输出的一个处理。通常输入是模型当前的输入tokens(input_ids)，以及对下一个token的预测的scores。这里的scores可能是模型最后一层的leaner 层的输出，可能是softmax的输出，还可能是其他类型的函数的输出。总之，这里就像是一个概率后处理一样的东西。

16. What is logits in LLMs?
在Transformers 库的 BART的源码中，logits其实就是模型 LM head 线性层的输出而已。并没有什么softmax。 softmax 的功能是 logits_processor 实现的。

17. d_model is the dimension of embeddings and also the dimension of hidden representations. It is not the max input sequence length.

18. Input sequence length is determined on the fly. Therefore, the input sequence length can very among batches. The input sequence length is arbitrary, determined by input.
基于这个事实，那么其实Encoder的隐藏层的输出序列长度其实可以和Decoder的输入序列长度不一致。这其实也是Encoder-Decoder架构在使用window attention进行生成时的情况。
并且Encoder 的输出矩阵是不断生长的。

19. Decoder的最后一个线形层的输入输出到底是什么？
输入为一个矩阵，即当前的输入序列对应的矩阵，但是输入维度定义为 d_model, 输出当然也是一个矩阵，只是矩阵的维度定义为vocabulary - 1。那么含义则是，对于每一个token，都会预测下一个token在vocabulary上的概率分布。
(Linear(in_features=1024, out_features=50264, bias=False), 50265)

20. Just curious, what will happen if we assign attention to paddings?
    https://huggingface.co/docs/transformers/en/glossary transformers术语表，transformers词汇表

21. positional wise FFN 原来是指这个FFN的输入只是一个token对应的embedding。这就是positional wise。并且这样的好处是能够处理变长的sequence。

22. feed forward chunking 是基于Transformer的positional wise特性。实现了时空trade-off。它是将并行计算转化为串行计算，从而增加计算时间，减少内存开销。这对于小内存的个人用户比较有用。 for `limited GPU memory`

23. image patch是指将一张图片拆分成一堆更小的图片。然后将这些图片视为sequence，输入给模型。就像NLP一样。

24. Three types of tasks related to texts:
Natural language generation (NLG)
All tasks related to generating text (for instance, Write With Transformers, translation).

Natural language processing (NLP)
A generic way to say “deal with texts”.

Natural language understanding (NLU)
All tasks related to understanding what is in a text (for instance classifying the whole text, individual words).

25. Pipeline is an end-to-end abstraction of the process of a task. It includes data preprocessing, model inference and data post-processing. Take language classification as an example, The input of a pipeline if raw text, the output of a pipeline is the labels and respective probabilities.

26. Abbreviations:
    DP & DDP: 
        DataParallel（DP）：Parameter Server模式，一张卡为reducer，实现也超级简单，一行代码。Parallelism technique for training on multiple GPUs where the same setup is replicated multiple times, with each instance receiving a distinct data slice.
        DistributedDataParallel（DDP）：All-Reduce模式，本意是用来分布式训练，但是也可用于单机多卡。
    PP:
        PipelineParallel (PP). Parallelism technique in which the model is split up vertically (layer-level) across multiple GPUs, so that only one or several layers of the model are placed on a single GPU.
    TP:
        Tensor Parallelism (TP). Parallelism technique for training on multiple GPUs in which each tensor is split up into multiple chunks.
    ZeRO：
        Zero Redundancy Optimizer (ZeRO). Parallelism technique which performs sharding of the tensors somewhat similar to TensorParallel (TP), except the whole tensor gets reconstructed in time for a forward or backward computation, therefore the model doesn’t need to be modified.
        Used to compensate for `limited GPU memory`.
    CTC: 
        connectionist temporal classification (CTC). An algorithm which allows a model to learn without knowing exactly how the input and output are aligned, e.g. speech recognition.
    MLM:
        masked language modeling (MLM). A pretraining task where the model sees a corrupted version of the texts, usually done by masking some tokens randomly, and has to predict the original text.

27. transfer learning
A technique that involves taking a pretrained model and adapting it to a dataset specific to your task. 和微调其实是同质的。

28. NER: 
    NER 是一个token wise的分类任务，类别包括 名称、地点、组织、数字等。其主要任务就是从文本中识别出这些entities。识别的目标是进行分析和匹配。它比直接使用关键词搜索的优势在于能够识别出不在数据集中的entities。并且它是批量处理的。基于识别的结果可以进行统计分析和匹配。这比直接分析token distribution更好。因为token只能分析token，不能分析phrases。我们可以根据entities的识别结果切分phrases。然后对切分结果进行分析。有时候往往是一个很小的区别导致了很大的结果差异，从而需要一个新的但是和原来方法相似的方法。长尾效应。
    Human resources: NER can speed up the hiring process by automatically filtering out resumes to find the appropriate candidates with the required skills. Specific skills can be used as entities for NER applications in hiring processes. 

29. Sentiment Analysis: Sentiment analysis is a Natural Language Processing (NLP) method that helps identify the emotions in text. 

30. Training dataset, validation dataset, test dataset, development dataset
验证集往往是训练集的一部分，比如 k-fold cross validation。 验证集的作用就是得到模型在训练集上的整体表现。
验证集又叫做 development dataset。
测试集则是用于测试摸性性能的数据集，比如得到f1-score，accuracy等指标。并且测试集还有一个重要的功能，就是衡量模型是否过拟合。

使用交叉验证的好处是，能知道一个模型框架在整个训练集上的性能表现，类似于将整个训练集都作为了测试集。这样能避免test dataset太过于固定和单一问题。
验证集往往是用于筛选模型超参的。




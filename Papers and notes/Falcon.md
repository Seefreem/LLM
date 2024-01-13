涉及到的两篇论文《The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only》
《Fast Transformer Decoding: One Write-Head is All You Need》
参考文章：https://huggingface.co/blog/falcon


论文中并没有讲模型部分。但是在hugging face上面可以找到对应的源码。

Falcon的一大工作重点在于数据集，它的团队精细化处理了RefinedWeb 网络数据。从而获取到更多的高质量数据。从而让模型更好。
第二个工作是，他们的模型使用了multi-query attention，也就是sharing keys and value embeddings across attention heads.
据说这种处理方法在不怎么影响预训练。但是在inference的时候却能够极大地减少内存开销。

Making inference with Falcon-7B only needs ~15GB. 所以将我的电脑的虚拟内存开大点。还是能跑跑7B的模型的。
在一个是，可以量化，量化之后，更能跑了。



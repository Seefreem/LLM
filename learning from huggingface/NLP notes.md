# pipeline
Pipeline 作为Huggingface的基础工具，能够很好地帮助我们直接使用huggingface上的模型。
当我们将自己的模型装进Pipeline后，也方便我们使用Huggingface的其他功能，比如 evaluator 。  
Pipeline is an abstract of the pre-process, modeling and post-process.

Some of the currently available pipelines are:  
- feature-extraction (get the vector representation of a text)
- fill-mask
- ner (named entity recognition)
- question-answering
- sentiment-analysis
- summarization
- text-generation
- translation
- zero-shot-classification
这些也是创建pipeline时需要输入的关键字。

`Zero-shot classification` 比较有意思。它是指在不经过微调的情况下，给text打上新的标签。比如 candidate_labels=["education", "politics", "business"]。这样的好处是，不用人为标注数据，加快数据生成和收集的过程。当然最好还是人为审核一下。

对于 `Text generation` 你可以控制生成多少个备选句子（num_return_sequences），还可以设置生成的句子的长度（max_length，整个输出句子的token数）。
在创建pipeline的时候也可以通过关键字model 指定模型。  
其实 pipeline 作为整个流程的抽象，那么它也是支持整个流程中所需要的参数的。 然后因为这层抽象可能包含错误的输入。所以在子模块中肯定是有鲁棒性检测的。

通过 Inference API 所有的模型都可以在huggingface的网页上进行试用。

在mask filling例子中，<mask>关键字表示一个 mask token。但是不同的模型可能会使用不同的token，因此有必要检查一下。检查的方法很多，可以看例子，可以看原始论文。

每个不同的任务或者模型可能都会有它们特定的参数，所以使用前记得检查。

对于"question-answering"pipeline，他们执行的结果是从context 中抽取答案，而不是基于context生成答案。
Note that this pipeline works by extracting information from the provided context; it does not generate the answer.


# Transformer models
Broadly, they can be grouped into three categories:  
- GPT-like (also called auto-regressive Transformer models)
- BERT-like (also called auto-encoding Transformer models)
- BART/T5-like (also called sequence-to-sequence Transformer models)

This type of model develops a statistical understanding of the language it has been trained on, but it’s not very useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called transfer learning. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.

An example of a task is predicting the next word in a sentence having read the n previous words. This is called causal language modeling because the output depends on the past and present inputs, but not the future ones.  

Each of these parts can be used independently, depending on the task:

- Encoder-only models: Good for tasks that require `understanding` of the input, such as sentence classification and named entity recognition. -- Gemini
- Decoder-only models: Good for `generative` tasks such as text generation. --GPT-x
- Encoder-decoder models or `sequence-to-sequence` models: Good for `generative` tasks that require an input, such as translation or summarization.

在训练s2s模型的时候，对于decoder来说还是一个无监督的学习，因为它还是在预测下一个token。  
但是我们可能会将这个训练理解为监督学习，因为我们提供了两个sequences。  
To speed things up during training (when the model has access to target sentences), the decoder is fed the whole target, but it is not allowed to use future words

Terminology:
- Architecture: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
- Checkpoints: These are the weights that will be loaded in a given architecture.
- Model: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify architecture or checkpoint when it matters to reduce ambiguity.

在Sequence-2-sequence 的例子中，他们也提到了，你可以在这个architecture中加载不同的ENcoder和Decoder来满足你的特定的任务需求。

Summary:  
Model	            Examples	                                Tasks
Encoder	            ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa	Sentence classification, named entity recognition,          
                                                                extractive question answering
Decoder	            CTRL, GPT, GPT-2, Transformer XL	        Text generation
Encoder-decoder     BART, T5, Marian, mBART	                    Summarization, translation, generative question answering

# behind the pipeline
```python 
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
``` 
这里我们通过 return_tensors 指定返回什么类型的 tensor， pt 指的是pytorch。
Don’t worry about padding and truncation just yet; we’ll explain those later. The main things to remember here are that you can pass one sentence or a list of sentences, as well as specifying the type of tensors you want to get back (if no type is passed, you will get a list of lists as a result).

python比较好的一点是，它具有非常丰富的库，能够实现很多功能，并且简单易上手（因为它能自动处理很多事情）。  
但是这也是python的麻烦之处。丰富+自动意味着隐藏的复杂度很高，很多时候产生的结果并不是你想要的。并且你还不知道。然后很多时候你需要注意类型，因为类型的不同会导致很多的错误。  

需要注意的是，往往模型在发布的时候都是移除了task-specific head，因此你在使用的时候要首先判断你下载的模型是否包含了head。  
怎么看？  
如果有对应的模型的定义，那么直接看源码，如果没有，那么就看看model card，看示例。如果还没有，那么就可以直接打印模型的结构（也就是直接print模型），再不行就直接看模型的输出。  
一般来说，没有head的名的输出格式是 [batch, sequence length, model dimension]

一般来说，transformer模型都是包含了embedding model的。你不用刻意去执行embedding，模型会自动计算。   

为了能方便的加载模型并且应用于不同的任务，Transformers提供了针对不同任务的抽象类： 
* Model (retrieve the hidden states)
* ForCausalLM
* ForMaskedLM
* ForMultipleChoice
* ForQuestionAnswering
* ForSequenceClassification
* ForTokenClassification

以 AutoModelForSequenceClassification 为例子，加载的模型中多了一层线性全连接层，但是注意全连接层的输出是logits，并且这个全连接层是线性的，不带激活函数。这里比较不一样的是，head的最后一层往往都不带激活函数。


还有需要注意的是在Transformers这个库中，模型的输出往往都是一个类似于字典的数据结构，因此你往往是通过关键词进行访问数据。    
Note that the outputs of 🤗 Transformers models behave like namedtuples or dictionaries. You can access the elements by attributes (like we did) or by key (outputs["last_hidden_state"]), or even by index if you know exactly where the thing you are looking for is (outputs[0]).

## about model
模型一般具有两个重要文件： configuration + checkpoint  
当我们执行 from_pretrained("bert-base-cased") 的时候，实际上执行的是先根据 configuration 文件创建 config 对象，然后创建模型的类，并且实例化，最后就是加载预训练的参数。  
这个过程可以我们自己写，可以打断。我们也可以通过修改配置文件来修改模型架构，然后从头开始训练。  

创建一个模型对象有两种方式，AutoModel 或者 特定的模型名。  
通过特定模型名创建模型的例子：
```python
from transformers import BertConfig, BertModel
config = BertConfig()
model = BertModel(config) # Model is randomly initialized!

# 或者
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased") # 等效于 AutoModel.from_pretrained("bert-base-cased")

```

The weights have been downloaded and cached (so future calls to the from_pretrained() method won’t re-download them) in the cache folder, which defaults to `~/.cache/huggingface/transformers`. You can customize your cache folder by setting the `HF_HOME` environment variable.

The `pytorch_model.bin` file is known as the state dictionary; it contains all your model’s weights. The two files go hand in hand; the `configuration` is necessary to know your model’s architecture, while the model weights are your model’s parameters.

## tokenizer
注意：直接使用tokenizer(sentences, max_length=1024, return_tensors="pt")就好。不要自己去写这个过程，不要使用 .tokenize(sequence) 这些子函数。因为像添加特殊token、padding和truncation 等功能都没有在子函数中实现。因此直接使用子函数肯定是有问题的。  
tokenization 的方法有很多，主要可以分为：word-based、Character-based 和 Subword tokenization。
Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word.

tokenizer的使用办法也挺多的。
你可以直接一步到位：
```python
doc1_inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")
```
也可以一步一步求解：
```python
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
```

```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
# =====================================================================
# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```
```python
model_inputs = tokenizer(sequences, padding="max_length", max_length=4, truncation=True)
```
对于上面的这行代码，如果字符串长度超过了 max_length，那么就只会保留前面的那部分，而直接丢弃后面的部分。这个隐藏的问题可能会导致意想不到的结果。


## processing the data--tokenization, batching and loading
对于像BERT这种能判断两个句子之间的关系的模型。它的tokenizer 可以接受两个字符串作为输入数据：
```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")
```
当然这两个句子也可以是两个等长的字符串列表。
tokenizer会将这两个字符串连接起来，并添加对应的特殊字符。
但是需要注意的是，这时候并不会进行padding。如果模型需要token_type_ids 的话，tokenizer也会自动生成对应的 token_type_ids。
因为上面没有指定 padding， 所以，可能并不会进行padding，这时候padding可能并不是最好的选择。因为这时候会选择最长的句子作为padding的标准。
padding可以在输入给模型的时候动态进行。

当数据集比较大的时候，将所有数据加载到内存中并进行处理可能会遇到内存大小瓶颈。因此使用 Dataset.map()  可以有效较少内存的需求。

It takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs) and will do everything you need:
```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```
注意这里在输入数据给data collator的时候，数据得是数字，不能是字符串什么的。并且这一步是基于tokenize之后的。
注意 data collator的作用是给输入的batch 自动添加 padding。而不是自动生成batch
```python
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
```
但是啊，他这里还是没讲怎么构建基于 bath 的 data loader。可能还是得回到pytorch。 或者得看看huggingface的 datadict 类。

Note that when you pass the tokenizer as we did here, the default data_collator used by the Trainer will be a DataCollatorWithPadding as defined previously, so you can skip the line data_collator=data_collator in this call. It was still important to show you this part of the processing in section 2!
又来了，又有一些默认的处理。
并且在写整体的Trainer的代码的时候，并没有移除多余的属性。
并且这里面似乎忽略了batch 的问题。或者说，并没有提供设置batch的选项。
所以这里有多种解决方式，第一是继续沿用Trainer，但是自己实现模型的推理函数。第二是自己构建训练过程。
在后面讲了，实际上还是采用了torch的dataloader


训练好的模型可以通过 .predict() 函数进行预测。得到的结果是字典，包含了 predictions, label_ids, and metrics 等信息。但请注意， all Transformer models return logits。所以拿到输出后还得进行转换。

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names
```
这里提到了几个关于 datasetdict的几个常用的函数。
然后基于上面的结果，通过 torch 的dataloader 来batch化 (但是手册又说，这些实际上都在Trainer中实现了，这听起来比较合理。可能还得看看源码)：
其实按照
```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

```
All 🤗 Transformers models will return the loss when labels are provided. 这挺好。

“A full training” 这一章很重要。讲了很多有用的知识。


# Dataset
load_dataset() 支持加载本地的多种文件。并且支持一次性加载多个文件，返回DatasetDict 对象。
还支持解压gzip, ZIP and TAR 文件。
还支持通过URL加载数据（streaming技术）。
The data_files argument of the load_dataset() function is quite flexible and can be either a single file path, a list of file paths, or a dictionary that maps split names to file paths. You can also glob files that match a specified pattern according to the rules used by the Unix shell (e.g., you can glob all the JSON files in a directory as a single split by setting data_files="*.json"). See the 🤗 Datasets documentation for more details.

Now that we’ve got a dataset to play with, let’s get our hands dirty with various `data-wrangling` techniques!

重磅！！！随机获取一小部分数据用于构建pipeline的方法如下：
```python
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
drug_sample[:3]

```
这里涉及到datasetDict 类的两个功能：shuffle和select（乱序和抽取）（slicing and dicing）
Dataset.select() expects an iterable of indices, so we’ve passed range(1000) to grab the first 1,000 examples from the shuffled dataset.
From this sample we can already see a few `quirks` in our dataset。

Dataset.unique(column name)  返回去重之后的元素列表。

new_dataset = DatasetDict.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
这个方法可以用于重命名 列。但是注意需要用一个新的对象来接住返回值。

通用 通过 .map() 函数，能够实现各种数据修改操作。

Dataset.filter() 函数用于筛选行数据。这个函数和map()函数的运行方式类似，只不过一次只支持一个样本。并且接受的函数的返回值不再是字典，而是boolean value。
常见的用法是和lambda函数结合：  
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None) # 有些数据行是空的，这里用于取掉空行。

经验：
Whenever you’re dealing with customer reviews, a good practice is to check the number of words in each review. A review might be just a single word like “Great!” or a `full-blown` essay with thousands of words, and depending on the use case you’ll need to handle these extremes differently. To compute the number of words in each review, we’ll use a rough `heuristic` based on splitting each text by whitespace.
Let’s define a simple function that counts the number of words in each review:

```python
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
```
通过返回新的 key的方式，可以创建新的列，并且实现赋值。另外还可以使用 Dataset.add_column() 添加新的列，用法和pandas 一样。

drug_dataset["train"].sort("review_length")[:3]
这个sort 函数能对列中的所有元素进行排序。但是这里有个问题，这个排序是in-place排序还是返回一个新的对象？


The last thing we need to deal with is the presence of HTML character codes in our reviews. We can use Python’s html module to unescape these characters, like so:
```python
import html

text = "I&#039;m a transformer called BERT"
html.unescape(text)
Copied
"I'm a transformer called BERT"
# We’ll use Dataset.map() to unescape all the HTML characters in our corpus:

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
```

Dataset.map() 的batched 参数默认情况下的batch size 是1000. 加速方法是 list comprehension
可以通过 batch_size 参数改变batch 大小。只不过这个batch size是执行map的大小，而不是输入给模型的batch size

This means that using a fast tokenizer with the batched=True option is 30 times faster than its slow counterpart with no batching — this is truly amazing! That’s the main reason why fast tokenizers are the default when using AutoTokenizer (and why they are called “fast”).
关于tokenization的加速问题：首先是设置 batched=True，然后是使用 fast Tokenizer，但是并不是所有的tokenizer都有fast版本
如果 batched=True 则传递给回调函数的值是一个子表，也就是每个属性下的元素类型是list。

```python
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)
tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)
```
使用 据说也能加速，但是我在实践的过程中似乎并没有体会到。
In general, we don’t recommend using Python multiprocessing for fast tokenizers with batched=True.
实际使用过程中，设置batched=True 就够了。

Datasets is designed to be interoperable with libraries such as Pandas, NumPy, PyTorch, TensorFlow, and JAX. Let’s take a look at how this works.
Datasets 类能将数据转化为pandas、numpy等其他类。

设置类型转换的方式也很简单： Dataset.set_format('pandas')
This function only changes the output format of the dataset, so you can easily switch to another format without affecting the underlying data format, which is Apache Arrow.
然后就可以这样使用了：
drug_dataset["train"][:3]
妈呀，python 已经被玩得完全变了样了。类型是什么？不存在的。
🚨 Under the hood, Dataset.set_format() changes the return format for the dataset’s __getitem__() dunder method. This means that when we want to create a new object like train_df from a Dataset in the "pandas" format, we need to slice the whole dataset to obtain a pandas.DataFrame. You can verify for yourself that the type of drug_dataset["train"] is Dataset, irrespective of the output format.
但是既然返回的是pandas 对象，那么就能级联调用pandas的方法。

从pandas到datasets：
freq_dataset = Dataset.from_pandas(frequencies)

This `wraps up` our tour of the various preprocessing techniques available in 🤗 Datasets. To `round out` the section, let’s create a validation set to prepare the dataset for training a classifier on. Before doing so, we’ll reset the output format of drug_dataset from "pandas" to "arrow":
drug_dataset.reset_format() # 设置回默认的输出格式

意见划分数据，将数据划分成多份，划分测试集，划分验证集：
```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean
```

注意：Datasets will cache every downloaded dataset and the operations performed on it。着了可能出现坑。
使用数据前检查数据总是好的。

注意如果要保存为JSON文件的话，每个数据集需要单独保存为一个文件：
For the CSV and JSON formats, we have to store each split as a separate file. One way to do this is by iterating over the keys and values in the DatasetDict object。
And that’s it for our excursion into data wrangling with 🤗 Datasets! 

如果你想节省空间，那么可以在解压之后删除压缩文件：download_config

```python
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
```
这里的 split 表示只加载 'train' 这部分的数据。  
streaming=True 得到的则是一个 迭代器。
可以通过指定batched 来使用batch。
迭代器也有 .map() 函数

shuffle 流式数据：
```python
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))
```
这是只shuffle 10_000 个数据，batch的大小自己指定。 shuffle之后，还是按照迭代器的方式进行访问。

```python
# Skip the first 1,000 examples and include the rest in the training set
train_dataset = shuffled_dataset.skip(1000)
# Take the first 1,000 examples for the validation set
validation_dataset = shuffled_dataset.take(1000)
```
也有一些选择和丢弃的函数。

最后还说到可以通过 interleave_datasets() 函数来组合多个数据集。

很逗，居然通过这种方式来实现数据类型的转换：
issues_dataset.set_format("pandas")
df = issues_dataset[:] # 转为pandas数据类型

pandas 的 df.explode() 函数能实现将包含多个元素的单元格展开并且膨胀为多行。其他列的信息将被复制到多行中。

这里还有个神操作：
直接将模型放在 map函数中：
```python
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```
get_embeddings 是自定义的函数，包含了一个模型。
但是这里还是不能利用模型的batch能力。因为map 的batch实际上是list comprehension.

Datasets 自带一个数据检索的功能：
Datasets called a FAISS index. FAISS (short for Facebook AI Similarity Search) is a library that provides efficient algorithms to quickly search and cluster embedding vectors.
可以实现vector store的功能。

# Tokenizer

# Evaluation
注意用于训练的 loss function 和evaluation 函数是不一样的。
并且交叉熵可以用于多分类任务。自然语言处理中预测下一个token就是多分类任务。
https://zhuanlan.zhihu.com/p/56638625


# Metrics
需要注意的是，Metrics的输入是文本，并且是衡量模型的输出的性能的。而不是损失函数。  
注意其中的 Perplexity 是直接使用目标token的概率来进行衡量的。这也不是损失函数。 
https://blog.csdn.net/codename_cys/article/details/108654792

## ppl (perplexity)
马尔可夫假设。用每个单词被预测的概率来计算文本的连贯程度。 

## bleu (bilingual evaluation understudy)
核心思想都是比较生成文本与参考文本间的字符串重合度。  
这个指标是基于 n-gram 的。 
衡量的是生成的文本“击中”参考文本的概率。  

## rouge 
这也是基于n-gram 的。只是衡量的内容不一样。  
衡量的是参考文本“击中”生成文本的概率。  

## bleurt
这种方法比较 fancy，它通过一个模型来对生成文本和参考文本进行评估，计算文本之间的距离。  

# Huggingface Metrics

https://huggingface.co/docs/datasets/en/how_to_metrics
这两个文章讲解了怎么加载 metrics，怎么下载metrics，以及怎么定义和加载自己的metrics。  
并且这里指定了一个使用 metrics 的接口：
```python
import datasets
metric = datasets.load_metric('my_metric')
for model_input, gold_references in evaluation_dataset:
    model_predictions = model(model_inputs)
    metric.add_batch(predictions=model_predictions, references=gold_references)
final_score = metric.compute()
```
接口是：直接将模型的输出 和 数据中的 label 输入给 metric，至于怎么计算，以及需要什么样的后处理，都交给metric 自己完成。  
而在Huggingface中 模型的输出 默认是 logits。  
`那么现在就是需要确定，predictions and references 长什么样。 需要看例子。`  


从这个文章来看，其实 metrics 的计算方法需要两个输入参数 predictions and references。  
其实并没有规定是 logits 还是 text。  
所以可能还是比较灵活。  
但是默认情况下可能还是比较 predictions 还是 logits。 但是references 就不确定了。因为不同的任务的label 不一样。  

## Inputs of metrics
text, logits, dictionary, numerics, etc.

## Choosing a metric for your task
https://huggingface.co/docs/evaluate/en/choosing_a_metric
metrics 的来源主要有三种：
1. 一般的accuracy、precision、recall等。
2. 数据集自定义的 metrics，那么这就需要看数据集的源码。
3. 任务自定义的metrics，这就需要看其对应的定义和源码了。

## Metrics
从几个 metrics 的源码来看，不同的metrics接受的输入不同。有些是 text，有些是字典。  
并且metrics中会有一些数据预处理的代码。  

那么Huggingface 提供的 meteics呢？  
huggingface上的模型的输出都是包含在一个输出类型中的，这个类型是根据不同的任务而不同，但是都是字典。  
并且一般来说，模型的直接输出都是logits（包含在 返回类型中）  

Metrics的使用方式很多。可以放在 Trainer中使用。可以放在 evaluator中使用。也可以拿到模型的输出之后，单独使用。它就是一个函数，随便你在哪里调用。

并且metric的使用往往是和 Evaluator 结合起来的：  
https://huggingface.co/docs/evaluate/en/base_evaluator

总结：  
关于 metrics 的使用方法： 三种使用方法。但是如果要放在Trainer中自动调用的话，往往需要进行 wrapping。  
huggingface 内部默认的调用方法是直接将模型输出和数据集中的label直接输入给metrics函数：  
```python
for model_input, gold_references in evaluation_dataset:
    model_predictions = model(model_inputs)
    metric.add_batch(predictions=model_predictions, references=gold_references)
    final_score = metric.compute()
``` 
但是注意，在 Trainer 中，并不是直接将模型的输出输入给 metrics的。 而是将模型的输出中的 logits拿出来，加上从数据中提取的labels，一起输入给metrics。

也就是说，可能在不同的地方，默认的规则不一样。需要看源码确定。  
或者另一个简单的办法是，自己写一个metrics函数，然后在这个函数中打印输入参数。然后就知道应该怎么处理了。  

### Accuracy
https://huggingface.co/spaces/evaluate-metric/accuracy 
```python
>>> accuracy_metric = evaluate.load("accuracy")
>>> results = accuracy_metric.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0])
>>> print(results)
{'accuracy': 0.5}
```
这是准确率的用法。但是我们知道模型的输出并不是这种格式的。并且有时候得到的是文本。而不是数字。  
这时候就需要进行预处理。或者对文本进行数字化，比如tokenization 或者直接比较字符串。就和 accuracy 本身的代码一样，直接比较值。  

实际案例：
```python
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```
在这里我们对 accuracy 进行了包装。因为一般来说模型的输出是 logits，或者包含了 logits的 字典。  
这里的预处理是对logits 进行 greedy selection。获得标签。  

### Perplexity
https://huggingface.co/spaces/evaluate-metric/perplexity
它这里提供的例子很奇怪。  
metric 在使用时需要指定 model_id，但是并没有说支持传入模型对象。  
那么就意味着，你要是想测试自己的模型，你就要需要替换成自己的模型的名字。或者将自己的模型传到huggingface上。  
并且鉴于这种自己加载模型的设定。它不适用于Trainer。 

### Rouge
https://huggingface.co/spaces/evaluate-metric/rouge
```python
>>> rouge = evaluate.load('rouge')
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> results = rouge.compute(predictions=predictions,
...                         references=references)
>>> print(results)
{'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
```
可以看到这里的输入也是text。  


### GLUE
https://github.com/huggingface/datasets/blob/main/metrics/glue/glue.py
从 GLUE 的源码来看，其实对于 accuracy 之类的 metrics， 它们的实现还是很简单暴力的。  

huggingface example:
https://huggingface.co/spaces/evaluate-metric/bleu
```python
>>> predictions = ["hello there general kenobi","foo bar foobar"]
>>> references = [
...     ["hello there general kenobi"],
...     ["foo bar foobar"]
... ]
>>> bleu = evaluate.load("bleu")
>>> results = bleu.compute(predictions=predictions, references=references)
>>> print(results)
{'bleu': 1.0, 'precisions': [1.0, 1.0, 1.0, 1.0], 'brevity_penalty': 1.0, 'length_ratio': 1.0, 'translation_length': 7, 'reference_leng}
# results = bleu.compute(predictions=predictions, references=references, tokenizer=word_tokenize)

```
可以看到在huggingface中它的输入也还是text。还可以将 tokenizer 传给 metric。


###  SacreBLEU metric
https://huggingface.co/spaces/evaluate-metric/sacrebleu
这里给出了一个例子：
```python
>>> predictions = ["hello there general kenobi", "foo bar foobar"]
>>> references = [["hello there general kenobi", "hello there !"],
...                 ["foo bar foobar", "foo bar foobar"]]
>>> sacrebleu = evaluate.load("sacrebleu")
>>> results = sacrebleu.compute(predictions=predictions, 
...                             references=references)
>>> print(list(results.keys()))
['score', 'counts', 'totals', 'precisions', 'bp', 'sys_len', 'ref_len']
>>> print(round(results["score"], 1))
100.0
```
注意，这里 metric 的输入是text。但是在解释中，他又说的是：‘predictions (list of str): list of translations to score. Each translation should be tokenized into a list of tokens.’
也就是ids。 但这也不是模型的logits。  


### SQuAD metric
https://github.com/huggingface/datasets/blob/main/metrics/squad/squad.py
这里提供了一个参考例子。  
```python
>>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
>>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
>>> squad_metric = datasets.load_metric("squad")
>>> results = squad_metric.compute(predictions=predictions, references=references)
>>> print(results)
{'exact_match': 100.0, 'f1': 100.0}
```
在这个例子中，metric的输入竟然又是 字典。 这个很符合 huggingface model 的输出。  
同时可以看到的是，在metric 里面，对输入的字典进行了解析。然后才是计算。  


### BLEURT metric
https://github.com/huggingface/datasets/blob/main/metrics/bleurt/bleurt.py

```python
>>> predictions = ["hello there", "general kenobi"]
>>> references = ["hello there", "general kenobi"]
>>> bleurt = datasets.load_metric("bleurt")
>>> results = bleurt.compute(predictions=predictions, references=references)
>>> print([round(v, 2) for v in results["scores"]])
[1.03, 1.04]
```
在这个例子中，metric 接受的输入是text。  


宏平均（Marco Averaged）:
对所有类别的每一个统计指标值的算数平均值，分别称为宏精确率（Macro-Precision） ，宏召回率（Macro-Recall），宏F值（Macro-F Score）

macro ˈmækrəʊ
1. [ADJ]You use macro to indicate that something relates to a general area, rather than being detailed or specific. 宏观的












# CoNLL 2003
CoNLL-2003 is a `named entity recognition` dataset released as a part of CoNLL-2003 shared task: language-independent named entity recognition.
https://paperswithcode.com/dataset/conll-2003
https://huggingface.co/datasets/conll2003

Example:
{
    "chunk_tags": [11, 12, 12, 21, 13, 11, 11, 21, 13, 11, 12, 13, 11, 21, 22, 11, 12, 17, 11, 21, 17, 11, 12, 12, 21, 22, 22, 13, 11, 0],
    "id": "0",
    "ner_tags": [0, 3, 4, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pos_tags": [12, 22, 22, 38, 15, 22, 28, 38, 15, 16, 21, 35, 24, 35, 37, 16, 21, 15, 24, 41, 15, 16, 21, 21, 20, 37, 40, 35, 21, 7],
    "tokens": ["The", "European", "Commission", "said", "on", "Thursday", "it", "disagreed", "with", "German", "advice", "to", "consumers", "to", "shun", "British", "lamb", "until", "scientists", "determine", "whether", "mad", "cow", "disease", "can", "be", "transmitted", "to", "sheep", "."]
}


# Universal Dependencies
Universal Dependencies (UD) is a framework for consistent `annotation of grammar` (parts of speech, morphological features, and syntactic dependencies) across different human languages.
https://universaldependencies.org/  



# Penn Treebank
The English Penn Treebank (PTB) corpus is one of the most known and used corpus for the evaluation of models for sequence labelling. The task consists of annotating each word with its `Part-of-Speech tag`. 
https://paperswithcode.com/dataset/penn-treebank


# WebNLG
这个数据集包含关系三元组和对应的文本描述。所以可以用于关系抽取，也可以用于基于关系的文本生成（referring expression generation）
The WebNLG corpus comprises of sets of triplets describing facts (entities and relations between them) and the corresponding facts in form of natural language text. 
https://paperswithcode.com/dataset/webnlg
https://huggingface.co/datasets/web_nlg

Example:
Triplets: (John_E_Blaha birthDate 1942_08_26) (John_E_Blaha birthPlace San_Antonio) (John_E_Blaha occupation Fighter_pilot)
Fact: John E Blaha, born in San Antonio on 1942-08-26, worked as a fighter pilot


# LLM Evaluation datasets
## GAIA (General AI Assistants) (Multi-Task Model Evaluation) 
https://klu.ai/glossary/gaia-benchmark-eval
The General AI Assistants (GAIA) benchmark, rigorously tests AI systems' multitasking abilities across complex, real-world scenarios. It assesses accuracy and the AI's handling layered queries.
适用于文本生成（generation）、问答、查询、检索等。
这个数据集包含了466 个问题。这些问题比较难，往往需要查阅网站、进行推理、进行归纳总结等。因为这个数据集是衡量AI助手的能力的。并且注意，这不是训练集。这是测试集。这也是个多模态数据集，涉及到模型处理文本、图像和表格。问题的答案是以字符串的形式呈现的，答案唯一。衡量方式是quasi-exact match。
Example:
According to github, when was Regression added to the oldest closed numpy.polynomial issue that has the Regression label in MM/DD/YY?


## MMLU (Massive Multi-task Language Understanding) (Multi-Task Model Evaluation)
https://klu.ai/glossary/mmlu-eval
This benchmark measures how well LLMs can multitask by evaluating their performance on a variety of tasks, such as question answering, text classification, and document summarization.
适用于文本生成（generation）
这个数据集是全面的多任务自然语言测试集。涉及到57个任务/topics，比如历史、法律、基础数学、逻辑推理、总结等。这个数据集涵盖的知识面很广。适用于评估foundation models。同时这个数据集也是专门针对于zero-shot和few-shot的场景。
数据集包含 15,908 个问题。涵盖了57个subjects， splited into a few-shot development set, a validation set, and a test set.
Example:
...一段文本...
From the passage, one may infer that the English Parliament wished to argue that the Act of Supremacy would:

(A) give the English king a new position of authority
(B) give the position of head of the Church of England to Henry VIII
(C) establish Calvinism as the one true theology in England
(D) end various forms of corruption plaguing the Church in England


## MMMU (Massive Multi-discipline Multimodal Understanding and Reasoning) 
https://klu.ai/glossary/mmmu-eval 
This benchmark evaluates the proficiency of LLMs in understanding and generating responses across multiple modalities, including text, images, and audio. It assesses the models' ability to perform tasks like image captioning, audio transcription, and cross-modal question answering.
The benchmark includes over 11.5K questions that span 30 subjects and 183 subfields, comprising 30 highly heterogeneous image types such as diagrams, tables, charts, chemical structures, photographs, paintings, geometric shapes, and musical scores.
这个数据集包含了30个学科，183个子学科的college-level knowledge和expert-level reasoning problem，共11.5K questions，涵盖了大量的图表和符号。主要用于衡量机器人的感知、推理和知识面。 
这个数据集被划分成few-shot、validation和testdataset。
和MMLU的区别在于这是多模态的数据。MMLU是纯文本的数据。


## MT Bench (Multi-Turn Benchmark)
https://klu.ai/glossary/mt-bench-eval
This benchmark measures how LLMs engage in coherent, informative, and engaging conversations. It is designed to assess the conversation flow and instruction-following capabilities.
Chatbot Arena

这个数据集是用于评估聊天机器人的 the conversation flow and instruction-following capabilities。
但是注意，It's important to note that the Elo score reflects a model's performance on a comparative single response rather than a multi-turn conversation.
包含80个数据：The benchmark consists of 80 high-quality, multi-turn questions tailored to assess conversation flow and instruction-following capabilities.

并且这个数据集在不断地增加。


## AlpacaEval
https://github.com/tatsu-lab/alpaca_eval
https://opendatalab.com/docs/datasets/?channel=datasets
https://klu.ai/glossary/alpaca-eval
AlpacaEval is an automated benchmarking tool that evaluates the performance of LLMs in following instructions. It uses the AlpacaFarm dataset to measure models' ability to generate responses that align with human expectations, providing a rapid and cost-effective assessment of model capabilities.

AlpacaEval, along with MT-Bench, is one of the best LLM evaluations for    understanding the relative ranking   of LLMs compared to their peers (other LLMs). While not perfect, it provides an automated comparison.
首先这个数据集是用于评估聊天机器人的。但是它是single-turn的评估，也就是只进行一轮对话。
这个数据集包含了2.5k的高质量人类标注数据。
AlpacaEval calculates win-rates for models across a variety of tasks, including traditional NLP and instruction-tuning datasets, providing a comprehensive measure of model capabilities.

## HELM (Holistic Evaluation of Language Models)
https://klu.ai/glossary/helm-eval
HELM is a comprehensive benchmark that evaluates LLMs on a wide range of tasks, including text generation, translation, question answering, code generation, and commonsense reasoning.

Holistic Evaluation of Language Models (HELM) is a comprehensive benchmark framework designed to improve the transparency of language models (LMs) by taxonomizing the vast space of potential scenarios and metrics of interest for LMs. Developed by Stanford CRFM, HELM serves as a living benchmark for the community, continuously updated with new scenarios, metrics, and models.
包括了 文本生成、翻译、问答、代码生成和尝试推理。
这个数据集评估的是LM在各个情景下的能力，数据集中的情景和衡量指标都在不断增加。
这个数据集对于LLM的评估就像是我们做一个性格测评一样。不适用于在论文中评估模型。
但是这个数据集聚焦于让模型的能力透明化，也就是帮助用户感知模型的能力图谱。

## HellaSwag 
https://klu.ai/glossary/hellaswag-eval
HellaSwag is a challenging new benchmark in AI for understanding and predicting human behavior. It involves predicting the ending of an incomplete video or text narrative, requiring a deep understanding of the world and human behavior.

HellaSwag is an evaluation dataset designed for studying `grounded commonsense inference`(基于常识的推理), a task that is typically easy for humans but challenging for machines. HellaSwag is an `acronym` for Harder Endings(更难的结局), Longer contexts, and Low-shot Activities for Situations With `Adversarial` Generations.

Commonsense reasoning for everyday tasks

An evolving benchmark
这个数据集是用于衡量机器人的常识推理的。
这个数据集是用于评估LM预测下一个句子的能力的。包含了70k 个问题。每个问题都是以一个问题加4个选项的方式组织起来的。
The dataset consists of 70,000 multiple-choice questions about `grounded situations`, each with four answer choices. The questions come from two domains: ActivityNet and WikiHow. The correct answer is the real sentence for the next event, while the three incorrect answers are adversarially generated and human-verified to fool machines but not humans.

## GSM8k (Grade School Math 8k) 
https://klu.ai/glossary/GSM8K-eval
https://paperswithcode.com/dataset/gsm8k
GSM8K, or Grade School Math 8K, is a dataset of 8,500 high-quality, linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.

这个数据集包含了8.5k 高质量的小学数学题。所有题目都是纯文本的，但是确实语言描述丰富的（linguistically diverse）。数据集被划分成训练集（7.5k）和测试集（1k）。注意这个数据集是包含了训练集和测试集的。每个问题都包含了2-8个步骤，但是每一步都只涉及到基础的数学运算（basic arithmetic operations）。也就是不涉及高等数学。

## GLUE (General Language Understanding Evaluation) 
https://gluebenchmark.com/tasks
GLUE is a benchmark that focuses on evaluating LLMs on natural language understanding tasks, such as question answering, text classification, and document summarization. The benchmark consists of two sub-benchmarks: HellaSWAG and MRPC.

The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
这个数据集也是包含了训练集和测试集的。
但是要注意，这个数据集是用于训练和评估模型的语言理解能力，而不是语言生成或者对话能力，也不是常识推理。
全是分类任务

## SuperGLUE
https://klu.ai/glossary/superglue-eval
https://super.gluebenchmark.com/
https://huggingface.co/datasets/super_glue

SuperGLUE is an updated version of GLUE that includes more challenging tasks, providing a more thorough evaluation of LLMs' capabilities.
全是分类任务
这个数据集包含了训练集、验证集和没有标签的测试集，用于评估模型的语言理解能力，并且专注于语言理解，排除了领域知识（domain knowledge），并且对标的是college students。
It was developed as an evolution of the General Language Understanding Evaluation (GLUE) benchmark, with the aim of addressing some of its limitations and providing a more comprehensive evaluation of language understanding models.
这个数据集包含了8个基础任务（primary tasks）和2个诊断任务（diagnostic tasks）。
The tasks within SuperGLUE include Boolean Questions (BoolQ), CommitmentBank (CB), Choice of Plausible Alternatives (COPA), and Multi-Sentence Reading Comprehension (MultiRC), among others.
有意思的是，这个数据集并没有提供测试集的答案，而是要求大家将测试集的预测结果上传到他们的网站上。
The benchmark provides public training and development datasets, while testing data is hidden and only used to evaluate predictions submitted to the leaderboard. The leaderboard contains information about each submission, as well as the scores for the subtasks included within the SuperGLUE benchmark.


# LLM Training datasets
Large language model training
There are four steps to training large language models:

1. Data collection and preprocessing 
The first step is to gather the training data set, which is the resource that the LLM will be trained on. The data can come from various sources such as books, websites, articles, and open datasets. 

Popular public sources to find datasets are:
- torchtext.datasets: https://pytorch.org/text/stable/datasets.html#id23
- Tensorflow Models and Datasets: https://www.tensorflow.org/resources/models-datasets https://www.tensorflow.org/datasets/catalog/overview#all_datasets
- Hugging Face
- Kaggle
- Google Dataset Search
- Data.gov
- Wikipedia database

The data then needs to be cleaned and prepared for training. This may involve converting the dataset to lowercase, removing stop words, and tokenizing the text into sequences of tokens that make up the text. 
这其实是很关键的一步。数据就是护城河。并且从这里可以看出，大家并没有一个共同的训练集，并且训练集也没有固定的来源。
LLM的世界是我不管你用什么数据训练，我只提供一些评估数据。你自个儿训练完，到我这里评估一下就好。
所以自己的数据和官方提供的数据之外的数据就显得尤为重要了。这也解答了你当时参加竞赛时的那个疑问——为什么要用官方提供的数据之外的数据？

寻找数据其实比较简单，你直接到深度学习平台、竞赛平台或者对应的论文中找就行了。
关键是寻找什么样的数据来完成你的训练？这可能涉及到使用别人使用的数据，也可能使用你自己定义的数据。

## GSM8K —— Math
## GLUE —— Language understanding
## SuperGLUE —— Language understanding
## WikiText-103 and WikiText-2
https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

这些数据来自Wikipedia。
这个数据集适合长文本依赖的模型。 over 100 million tokens
Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger. The WikiText dataset also features a far larger vocabulary and retains the original case, punctuation and numbers - all of which are removed in PTB. As it is composed of full articles, the dataset is well suited for models that can take advantage of `long term dependencies`.

Each file contains `wiki.train.tokens`, `wiki.valid.tokens`, and `wiki.test.tokens`. No processing is needed other than replacing newlines with <eos> tokens.
包含了训练集、测试集和验证集。

## Toronto BookCorpus
https://en.wikipedia.org/wiki/BookCorpus
https://github.com/sgraaf/Replicate-Toronto-BookCorpus
这些数据来自Smashwords，大概7,000 本书，每本书大概包含20k tokens。
对应的论文： Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books
是训练LM的基本数据集。

## Gutenberg
https://www.gutenberg.org/
https://odonnell31.medium.com/nlp-in-python-a-primer-on-nltk-with-project-gutenberg-fcc02be63d9a
https://www.nltk.org/book/ch02.html
https://www.nltk.org/
Project Gutenberg is a library of over 70,000 free eBooks
这个数据集也是书籍。
通过 NLTK 库来使用。
也适合用于训练LM。

## Common Crawl
https://commoncrawl.org/
这是网页数据。
Over 250 billion pages spanning 17 years.

## Text8 Dataset
https://dhananjaytomar.medium.com/using-benchmark-datasets-character-level-language-modeling-ef16afa21101
数据来自Wiki，适合长文本建模。只有文本。没有XML标签。基本单元是characters。
Dataset homepage: http://mattmahoney.net/dc/textdata.html
Vocabulary: Lowercase English characters and space.
Training data: First 90M characters.
Validation data: First 5M characters out of the last 10M characters.
Testing data: Last 5M characters.

## Hutter Prize Dataset (aka enwik8)
https://dhananjaytomar.medium.com/using-benchmark-datasets-character-level-language-modeling-ef16afa21101
这个数据集包含了XML标签。适合长文本建模。但是这个数据集的基本数据单元是 bytes，而不是characters。
Dataset homepage: http://mattmahoney.net/dc/textdata.html
Vocabulary: 205 one-byte Unicode symbols which form 6064 characters.
Training data: First 90M characters.
Validation data: First 5M characters out of the last 10M characters.
Testing data: Last 5M characters.

# NLTK
https://www.nltk.org/

# Datasets used for training and evaluating GPT-1/2/3
## GPT-1 
### pre-training dataset
BooksCorpus dataset and Word Benchmark
- BooksCorpus 
就是上面的 BookCorpus

- Word Benchmark
1B Word Benchmark or Billion Word Benchmark or 1BW
The One Billion Word dataset is a dataset for language modeling. The training/held-out data was produced from the WMT 2011 News Crawl data using a combination of Bash shell and Perl scripts.


### fine-tuning and evaluation dataset 
Table 1: A list of the different tasks and datasets used in our experiments. (used in fine-tuning stage)
Task                            Datasets
Natural language inference      SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25]
Question Answering              RACE [30], Story Cloze [40]
Sentence similarity             MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6]
Classiﬁcation                   Stanford Sentiment Treebank-2 [54], CoLA
在这篇论文中，模型的训练分为两个部分 pre-training和fine-tuning。
pre-training 阶段训练的是LM。

- SNLI
https://huggingface.co/datasets/snli
https://nlp.stanford.edu/projects/snli/
The Stanford Natural Language `Inference` (SNLI) Corpus.
570k human-written English sentence pairs manually labeled
数据结构就是 两个句子 加 一个关系（entailment, contradiction, and neutral）
模型的任务就是判断两个句子之间的关系。其实就是个三分类的问题。
- MultiNLI
MNLI
https://cims.nyu.edu/~sbowman/multinli/
https://huggingface.co/datasets/multi_nli
The Multi-Genre Natural Language Inference (MultiNLI) corpus
433k sentence pairs annotated with textual entailment information
The corpus is modeled on the SNLI corpus, but differs in that covers a range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation. 
也是三分类。
- SQuAD (Stanford Question Answering Dataset)
https://paperswithcode.com/dataset/squad
QA类型的数据集。
more than 100,000 questions 
属于 passage + question + answer的类型

- Question NLI
https://paperswithcode.com/dataset/qnli
首先这个数据集来自Stanford Question Answering Dataset v1.1 (SQuAD)，但是做了些修改，原来的任务是从文本中提取出问题的答案。但是在这个数据集中，仅仅判断参考文本中是否存在问题的答案。所以可能有答案，可能没有。
其次，这个数据集是GLUE dataset的一部分。
The task is to determine whether the context sentence contains the answer to the question.
所以这其实是个二分类问题。
- RTE
https://paperswithcode.com/dataset/rte
The Recognizing Textual Entailment (RTE) datasets come from a series of textual entailment challenges. Data from RTE1, RTE2, RTE3 and RTE5 is combined. Examples are constructed based on news and Wikipedia text.
- SciTail
https://allenai.org/data/scitail
https://paperswithcode.com/dataset/scitail

这个数据集来自science exams。包含了很多的问题，但是用提供问题和答案的陈述性表达。
The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. Each question and the correct answer choice are converted into an assertive statement to form the hypothesis.
- RACE
https://www.cs.cmu.edu/~glai1/data/race/
https://paperswithcode.com/dataset/race
这个数据及比较有意思。这个数据集是中国人做的。数据来源是中国的初高中英语考试。
包含了28,000篇文章和100,000个问题。
Race is a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students. The dataset can be served as the training and test sets for machine comprehension.

Data Usage:
Each passage is a JSON file. The JSON file contains following fields:

article: A string, which is the passage.
questions: A string list. Each string is a query. We have two types of questions. First one is an interrogative sentence. Another one has a placeholder, which is represented by _.
options: A list of the options list. Each options list contains 4 strings, which are the candidate option.
answers: A list contains the golden label of each query.
id: Each passage has a unique id in this dataset.
很逗，这不就是直接将英语阅读题给做成了JSON文件吗？
- Story Cloze
https://paperswithcode.com/dataset/storycloze
https://cs.rochester.edu/nlp/rocstories/
这个数据集是关于故事的。给定故事的主题，让模型判断一个句子是否是这个故事的结局。可以被用于QA，也可以用于NLI，还可以用于generation。
'Story Cloze Test' is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning. 
This test requires a system to choose the correct ending to a four-sentence story. (可以组织成分类问题)
To enable the Story Cloze Test, we created a new corpus of five-sentence commonsense stories, 'ROCStories'. 
- MSR Paraphrase Corpus
https://www.microsoft.com/en-us/download/details.aspx?id=52398
https://autonlp.ai/datasets/microsoft-research-paraphrase-corpus-(mrpc)
Microsoft Research Paraphrase Corpus (MRPC)
这是个比较早的数据集了，发布于2005年，其时效性可能得引起重视。
包含了5,800条文本对。
模型需要判断两个句子是否在语义上对等，也就是判断是否是转述。
a text file containing 5800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship. Last published: March 3, 2005.

- Quora Question Pairs
https://paperswithcode.com/dataset/quora-question-pairs
https://www.kaggle.com/c/quora-question-pairs
比较有趣的是，这个数据集竟然不是问答，而是转述。
Quora Question Pairs (QQP) dataset consists of over 400,000 question pairs, and each question pair is annotated with a binary value indicating whether the two questions are paraphrase of each other.

- STS Benchmark (Semantic Textual Similarity)
https://paperswithcode.com/dataset/sts-benchmark
衡量的是文本相似度。
STS Benchmark comprises a selection of the English datasets used in the STS tasks organized in the context of SemEval between 2012 and 2017. The selection of datasets include text from image captions, news headlines and user forums.

- Stanford Sentiment Treebank
https://towardsdatascience.com/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065
https://paperswithcode.com/dataset/sst
这个数据集是情感分类的。数据来源于电影评价。每个句子都给解析成 解析树。
包含了11,855 single sentences和215,154 unique phrases。

- CoLA
https://nyu-mll.github.io/CoLA/
这个数据集是判断句子的语法可接受度的。属于二分类问题——接受和不接受。
The Corpus of Linguistic Acceptability (CoLA) in its full form consists of 10657 sentences from 23 linguistics publications, expertly annotated for acceptability (grammaticality) by their original authors. The public version provided here contains 9594 sentences belonging to training and development sets, and excludes 1063 sentences belonging to a held out test set. 

## GPT-2
### pre-training
Based on Common Crawl, they created a new dataset, WebText.
WebText contains slightly over 8 million documents for a total of 40 GB of text. They removed all Wikipedia documents from WebTex to avoid overlapping with evaluation datasets.

应该也是包含了GPT-1中使用到的那些数据。
https://paperswithcode.com/dataset/webtext
https://huggingface.co/datasets/Skylion007/openwebtext


### Zero-shot evaluation
LAMBADA, CBT-CN, CBT-NE, WikiText2, PTB, enwik8, text8, WikiText103, 1BW

- LAMBADA
https://paperswithcode.com/dataset/lambada
https://arxiv.org/abs/1606.06031
首先这是个测试集。不是训练集。
其次这测试集是用于测试模型预测下一个单词的能力的。还测试模型的长文本理解能力。

- CBT
https://paperswithcode.com/dataset/cbt
The Children’s Book Test (CBT) is designed to measure directly how well language models can exploit wider linguistic context. The CBT is built from books that are freely available.  
The CBT is built from books that are freely available thanks to Project Gutenberg.
那么要注意训练集和测试集不能重复。
This dataset contains four different configurations:  
V: where the answers to the questions are verbs.
P: where the answers to the questions are pronouns.
NE: where the answers to the questions are named entities.
CN: where the answers to the questions are common nouns.




## GPT-3
### ptr-training
Dataset                 Quantity(tokens)    Weight in training mix      Epochs elapsed when training for 300B tokens
Common Crawl (ﬁltered)  410 billion         60%                         0.44
WebText2                19 billion          22%                         2.9
Books1                  12 billion          8%                          1.9
Books2                  55 billion          8%                          0.43
Wikipedia               3 billion           3%                          3.4

Table 2.2: Datasets used to train GPT-3
他这里训练模型的时候使用了一个trick，就是上面的百分比并不是各种数据集大小的百分比，而是每次采样时的百分比，也就是从每个数据集中采集的样本占这个batch的百分比。
这其实就是在组合信息。尽量保证占比最大的信息占到主导地位，不被后来的数据给冲掉。


### evaluation


# Datasets for Evaluating Gemini
Capability	    Benchmark 	    Description
General	        MMLU	        Representation of questions in 57 subjects (incl. STEM, humanities, and others)
Reasoning	    Big-Bench Hard	Diverse set of challenging tasks requiring multi-step reasoning
                DROP	        Reading comprehension (F1 Score)
                HellaSwag	    Commonsense reasoning for everyday tasks
Math	        GSM8K	        Basic arithmetic manipulations (incl. Grade School math problems)
                MATH	        Challenging math problems (incl. algebra, geometry, pre-calculus, and others)
Code	        HumanEval	    Python code generation
                Natural2Code	Python code generation. New held out dataset HumanEval-like, not leaked on the web

- BIG-Bench
https://paperswithcode.com/dataset/big-bench
https://github.com/google/BIG-bench?tab=readme-ov-file#quick-start-colab-notebooks
https://huggingface.co/datasets/bigbench
https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/keywords_to_tasks.md

The Beyond the Imitation Game Benchmark (BIG-bench) is a collaborative benchmark intended to probe large language models and extrapolate their future capabilities. Big-bench include more than `200 tasks`.
The benchmark contains over 204 language-related tasks, from chess-based prompts to emoji-guessing tasks.

任务类型：
traditional NLP tasks	
logic, math, code	
understanding the world	
understanding humans	
scientific and technical understanding	
mechanics of interaction with model	
targeting common language model technical limitations	
pro-social behavior	
other	
据说都是reasoning 类型的任务

- Big-Bench Hard
https://arxiv.org/abs/2210.09261
https://github.com/suzgunmirac/BIG-Bench-Hard

This dataset focuses on a suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH). These are the task for which prior language model evaluations did not outperform the average human-rater. 

- DROP
https://paperswithcode.com/dataset/drop
https://allenai.org/data/drop
https://arxiv.org/abs/1903.00161

这个数据集是一个reasoning数据集。包含了训练集、development dataset和test dataset。
这个数据集处理的是段落信息抽取和推理。
这个数据的形式是 passage + question + answer。就很适合做参考学习（你想做的那个）。

Discrete Reasoning Over Paragraphs DROP contains 96k-question, in which a system must resolve references in a question, perhaps to multiple input positions, and perform discrete operations over them (such as addition, counting, or sorting). 
The questions consist of passages extracted from Wikipedia articles. 
The dataset is split into a training set of about 77,000 questions, a development set of around 9,500 questions and a hidden test set similar in size to the development set.

- MATH
https://arxiv.org/pdf/2103.03874.pdf
https://github.com/hendrycks/math?tab=readme-ov-file
https://paperswithcode.com/dataset/math
这就是个纯数学数据集
MATH is a new dataset of 12,500 challenging competition mathematics problems. 
Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations.

- HumanEval
https://arxiv.org/abs/2107.03374
https://github.com/openai/human-eval?tab=readme-ov-file
https://huggingface.co/datasets/openai_humaneval
https://paperswithcode.com/dataset/humaneval

这是一个a new evaluation set。用于评估模型写代码的能力。
预训练是 github上可用的代码。

This is an evaluation harness for the HumanEval problem solving dataset described in the paper "Evaluating Large Language Models Trained on Code". It used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.


# Datasets for BERT










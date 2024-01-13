# Requires:
# pip install langchain docarray

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import DocArrayInMemorySearch # 这就是向量数据库部分，内存版本
api_key = "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"

# -------------------设置环境变量-------------------------
import os
os.environ["OPENAI_API_KEY"] = "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"

#### Step 1: Build a vectorstore and get the retriever
print('Setup vector database')
# 给向量数据库增加条目
vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings(),
)
# Retriever does three things: 1. embedding input string; 2. Relative retrieving; 3. decoding embeddings. 
retriever = vectorstore.as_retriever() # 获取到向量数据库的检索器，检索器能够自动根据输入检索出相关的信息。
print(retriever.invoke("where did harrison work?"))

#### Step 2: Set up prompt template
print('Setup template')
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

#### Step 3: Initialize a model
model = ChatOpenAI()
output_parser = StrOutputParser()

#### Step 4: Set up the chain and play with it
# 下面这部分是创建的一个 chain，其中第一步的RunnableParallel 是将用户输入和向量数据库组合起来的函数。
# context 的内容由 retriever 填充（调用的返回值）。question 部分由 RunnablePassthrough (用户输入)填充。
# 这里省略了一个环节，那就是根据用户输入寻找相关信息的算法是被省略了的。这个算法是retriever内部实现了的，它也是一个研究热点。
# RunnableParallel 创建一个控模型，属于LCEL，模型内容被定义为一个字典。运行结果也是这个字典，然后将字典返回。
# 然后这个字典被传送给prompt的invoke函数。
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
# 理论上能够继承Langchain 的 pipeline 对象，然后覆盖其invoke方法，就能实现自定义的输出节点了。invoke的输入和输出都是一样的。
# 只是中间添加了对应的打印语句。
# LCEL 就是简化了一系列的嵌套的invoke()函数。 LECL transforms a series of invokes into a nested stacked invoke.  
chain = setup_and_retrieval | prompt | model | output_parser

print(chain.invoke("where did harrison work?"))

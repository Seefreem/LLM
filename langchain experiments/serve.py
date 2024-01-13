#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes
# 这个例子是启动一个server，但是我还没搞懂它是怎么运行的。以及怎么运用到我的开发中。

# 1. Chain definition

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""


    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who generates comma separated lists.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
api_key = "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"
category_chain = chat_prompt | ChatOpenAI(openai_api_key=api_key) | CommaSeparatedListOutputParser()


# 2. App definition, 这里就定义好了一个server，能自动处理请求和返回结果
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 3. Adding chain route
# 这里就像是配置server，配置处理请求的函数以及server的工作空间。
add_routes(
    app, # server
    category_chain,  # 处理请求的函数
    path="/category_chain", # 工作空间
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
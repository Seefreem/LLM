from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel
from langchain.schema.output_parser import StrOutputParser
import os
# Set api_key. 设置api_key
os.environ["OPENAI_API_KEY"] = ""
output_parser = StrOutputParser()
model = ChatOpenAI()
joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model | output_parser
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model | output_parser
)
# 注意参数部分参数名会被用于生成结果字典中的key。
# 生成并行链，类似于多线程。
map_chain = RunnableParallel(jokes=joke_chain, poem=poem_chain)
response = map_chain.invoke({'topic': 'bear'})
print(type(response), response['jokes'], '\n', response['poem'])





# https://python.langchain.com/docs/expression_language/get_started
# Basic example: prompt + model + output parser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
# Set api_key. 添加 api_key
api_key = ''

prompt = ChatPromptTemplate.from_template('Tell me a joke about {topic}')
llm = ChatOpenAI(openai_api_key = api_key)
output_parser = StrOutputParser()
# 实例化prompt的方法： 并且注意，这个prompt类默认输出message，但是能够自动识别LLM 和 ChatModel。从而自动转化为Message或者string。
print('Prompt:', prompt.format(topic= 'ice cream'))
print('Prompt:', prompt.invoke({'topic': 'ice cream'}))
# prompt.invoke({'topic': 'ice cream'}).to_messages()
# prompt.invoke({'topic': 'ice cream'}).to_string()

respond = llm.invoke(prompt.format(topic= 'ice cream'))
# 逐步查看输出
print(respond)
print(output_parser.invoke(respond))
chain = prompt | llm | output_parser
print(chain.invoke({'topic' : 'ice cream'}))

# # --------------- 创建LLM 对象---------------
# from langchain.llms import OpenAI
# llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key = api_key)
# print(llm.invoke(prompt.invoke({'topic': 'ice cream'}).to_string()))

# 在这篇文章中，它指出，你如果想要查看中间结果，那么你就可以在要查看结果的那个组件那里断开链，增加一个输出。但是能不能将输出也作为一个组件？



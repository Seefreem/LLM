

from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os
os.environ["OPENAI_API_KEY"] = "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"

# prompt = ChatPromptTemplate.from_template(
#     "Tell me a short joke about {topic}"
# )
# output_parser = StrOutputParser()
# model = ChatOpenAI(model="gpt-3.5-turbo")
# chain = (
#     {"topic": RunnablePassthrough()} 
#     | prompt
#     | model
#     | output_parser
# )

# chain.invoke("ice cream")

# for chunk in chain.stream("ice cream"):
#     print(chunk, end="", flush=True)

# print('batch:')
# # Streaming 和直接invoke的区别在于，OpenAI是streaming的。invoke是等到结果完整了才输出。但是streaming模式可以收到啥打印啥。
# # print(chain.batch(["ice cream", "spaghetti", "dumplings"])) # 使用Batch 还有频率限制。得充钱



from langchain.llms import OpenAI
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
output_parser = StrOutputParser()

llm = OpenAI(model="gpt-3.5-turbo-instruct")
llm_chain = (
    {"topic": RunnablePassthrough()} 
    | prompt
    | llm
    | output_parser
)

print(llm_chain.invoke("ice cream"))



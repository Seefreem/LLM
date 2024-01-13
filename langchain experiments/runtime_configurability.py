from langchain_core.runnables import ConfigurableField
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
os.environ["OPENAI_API_KEY"] = ""

prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
model = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# --------------- openai----------------
from langchain.llms import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instruct")
# --------------- ChatAnthropic---------
from langchain.chat_models import ChatAnthropic
anthropic = ChatAnthropic(model="claude-2")
# 将多个模型放在同一个chain节点上。组装多个模型。联合使用多个模型。
# 这样的好处是能够将多个微调模型组合起来，完成更多的任务。
# 使用像配置一样的方式去调用不同的模型。
# 注意前面是分别定义对应的LLMs。下面是注册到对应的容器中。
configurable_model = model.configurable_alternatives(
    ConfigurableField(id="model"), # 定义可配置参数的参数名
    default_key="chat_openai", 
    openai=llm, # 键值对，左边的是key。key是可配置的参数的值，这里是model的值
    anthropic=anthropic, # 键值对，左边的是key
)
configurable_chain = (
    {"topic": RunnablePassthrough()} 
    | prompt 
    | configurable_model 
    | output_parser
)

result = configurable_chain.invoke(
    "ice cream", 
    config={"model": "openai"}
)
print('OpenAI:', result)

stream = configurable_chain.stream(
    "ice cream", 
    config={"model": "anthropic"}
)
print('Anthropic:')
for chunk in stream:
    print(chunk, end="", flush=True)

# configurable_chain.batch(["ice cream", "spaghetti", "dumplings"])
# await configurable_chain.ainvoke("ice cream")

'''
最后注意一下 Fallbacks 的功能就好。如果当前链调用失败了，那么就将同样的参数输入给 Fallbacks 指定的链。

def invoke_chain_with_fallback(topic: str) -> str:
    try:
        return invoke_chain(topic)
    except Exception:
        return invoke_anthropic_chain(topic)

async def ainvoke_chain_with_fallback(topic: str) -> str:
    try:
        return await ainvoke_chain(topic)
    except Exception:
        # Note: we haven't actually implemented this.
        return ainvoke_anthropic_chain(topic)

async def batch_chain_with_fallback(topics: List[str]) -> str:
    try:
        return batch_chain(topics)
    except Exception:
        # Note: we haven't actually implemented this.
        return batch_anthropic_chain(topics)

invoke_chain_with_fallback("ice cream")
# await ainvoke_chain_with_fallback("ice cream")
batch_chain_with_fallback(["ice cream", "spaghetti", "dumplings"]))
'''
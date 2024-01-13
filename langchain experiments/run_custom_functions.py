from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
import os
# Set api_key. 设置api_key
os.environ["OPENAI_API_KEY"] = "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"

def length_function(text):
    return len(text)


def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("what is {a} + {b}")
model = ChatOpenAI()

chain1 = prompt | model

# 从这里可以看到的是，chain之间传递的仅仅是一个值而已，也就是invoke嵌套调用的逻辑。
# 因此理论上chain是可以任意组合嵌套的。
# 并且一个chain节点就是一个表达式，表达式的值将被传递给下一个chain节点。所以chain节点其实不必是具有invoke的对象。
# 并且就像编程语言中的函数调用一样，在调用下一层函数之前，需要先求解出当前层表达式的值。
# 像这里的chain的第一一个节点，就只是一个单纯的字典表达式。
# 字典中的value部分也是chain，chain也是一个表达式。
# 并且定义chain就像是定义函数一样。调用chain就像是调用函数一样。
# 这些内容在 https://python.langchain.com/docs/expression_language/get_started 这里都有解释
# 使用 “Runnable” protocol.
# chain的本质还是函数调用，只是换了一种写法。
chain = (
    {   # itemgetter("foo") 表示获取输入列表中 foo 的值
        "a": itemgetter("foo") | RunnableLambda(length_function), # 这里是嵌套逻辑，"a"的值等于后面的chain的最终结果
        "b": {"text1": itemgetter("foo"), "text2": itemgetter("bar")}| RunnableLambda(multiple_length_function), # 这里也是嵌套逻辑
    }
    | prompt
    | model
)
print(chain.invoke({"foo": "bar", "bar": "gah"}))
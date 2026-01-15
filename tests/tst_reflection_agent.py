import sys
sys.path.append("..")
from core import OpenAICompatibleLLM
from agents import ReflectionAgent

client = ReflectionAgent(
    name="代码智能生成助手", 
    llm=OpenAICompatibleLLM()
    )

client.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)")
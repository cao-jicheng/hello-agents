import sys
sys.path.append("..")
from core import OpenAICompatibleLLM, ToolRegistry
from agents import ReActAgent
from tools import global_tool_registry

client = ReActAgent(
    name="AI智能助手", 
    llm=OpenAICompatibleLLM(),
    tool_registry=global_tool_registry
    )

client.run("根据成都当前的天气情况，推荐2~3个旅游景点，直接给出名字即可，不需要详细介绍")



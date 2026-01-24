from core import OpenAICompatibleLLM
from agents import ReActAgent
from tools import ToolRegistry, toolset_maps

registry = ToolRegistry()
for k, v in toolset_maps.items():
    registry.register_function(
    name=k,
    description=v["description"],
    func=v["func"]
)

client = ReActAgent(
    name="AI智能助手", 
    llm=OpenAICompatibleLLM(),
    tool_registry=registry
    )

client.run("根据成都当前的天气情况，推荐2~3个旅游景点，直接给出名字即可，不需要详细介绍")



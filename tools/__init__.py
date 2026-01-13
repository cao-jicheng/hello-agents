import sys
sys.path.append("..")
from core import ToolRegistry
from .calculator_tool import calculator
from .searcher_tool import searcher, summarized_searcher
from .weather_tool import get_weather

global_tool_registry = ToolRegistry()

global_tool_registry.register_function(
    name="calculator",
    description="简单的数学计算器，支持基本四则混合运算、常见数学函数运算",
    func=calculator
)
global_tool_registry.register_function(
    name="summarized_searcher",
    description="互联网搜索工具，整合Tavily和Bocha多个搜索源，提供更全面的搜索结果",
    func=summarized_searcher
)
global_tool_registry.register_function(
    name="get_weather",
    description="根据城市名，实时获取天气、温度、湿度、可见度等级信息。注意：城市名需要先转换成英文，且不包含市、省、区。例如：Chengdu",
    func=get_weather
)
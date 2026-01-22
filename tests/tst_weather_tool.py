import sys
sys.path.append("..")
from tools import ToolRegistry, toolset_maps

registry = ToolRegistry()
registry.register_function(
    name="get_weather",
    description=toolset_maps["get_weather"]["description"],
    func=toolset_maps["get_weather"]["func"]
)

test_cities = [
    "chengdu",
    "西安市",
    "<未知地>"
    ]

for i, city in enumerate(test_cities, 1):
    print('-' * 100)
    print(f"测试{i}: {city}")
    result = registry.execute_tool("get_weather", city)
    print(result)
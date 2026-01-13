import sys
sys.path.append("..")
from tools import global_tool_registry

test_cities = [
    "chengdu",
    "西安市",
    "<未知地>"
    ]

for i, city in enumerate(test_cities, 1):
    print('-' * 100)
    print(f"测试{i}: {city}")
    result = global_tool_registry.execute_tool("get_weather", city)
    print(result)
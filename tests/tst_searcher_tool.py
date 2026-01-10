import sys
sys.path.append("..")
from core import ToolRegistry
from tools import searcher

registry = ToolRegistry()
registry.register_function(
    name="advanced_searcher",
    description="高级搜索工具，整合Tavily和Bocha多个搜索源，提供更全面的搜索结果",
    func=searcher.search
)

test_queries = [
    "  ",
    "What is the latest status of AI",
    "成都市区最值得去哪一个旅游景点"
    ]

for i, query in enumerate(test_queries, 1):
    print('-' * 100)
    print(f"测试{i}: {query}")
    result = registry.execute_tool("advanced_searcher", query, auto_summary=True)
    print(result)

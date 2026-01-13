import sys
sys.path.append("..")
from tools import global_tool_registry

test_queries = [
    "  ",
    "What is the latest status of AI",
    "成都市区最值得去哪一个旅游景点"
    ]

for i, query in enumerate(test_queries, 1):
    print('-' * 100)
    print(f"测试{i}: {query}")
    result = global_tool_registry.execute_tool("summarized_searcher", query)
    print("=== AI提炼汇总后的结果 ===")
    print(result)

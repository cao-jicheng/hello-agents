from tools import ToolRegistry, toolset_maps

registry = ToolRegistry()
registry.register_function(
    name="summarized_searcher",
    description=toolset_maps["summarized_searcher"]["description"],
    func=toolset_maps["summarized_searcher"]["func"]
)

test_queries = [
    "  ",
    "What is the latest status of AI",
    "成都市区最值得去哪一个旅游景点"
    ]

for i, query in enumerate(test_queries, 1):
    print('-' * 100)
    print(f"测试{i}: {query}")
    result = registry.execute_tool("summarized_searcher", query)
    print("=== AI提炼汇总后的结果 ===")
    print(result)

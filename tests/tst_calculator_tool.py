from tools import ToolRegistry, toolset_maps

registry = ToolRegistry()
registry.register_function(
    name="calculator",
    description=toolset_maps["calculator"]["description"],
    func=toolset_maps["calculator"]["func"]
)

test_cases = [
    "2 + 3", 
    "9 - 4",       
    "5 * 7",        
    "15 / 6",
    "1 / 0",
    "sqrt(16)",   
]

for i, expression in enumerate(test_cases, 1):
    print(f"测试{i}: {expression}")
    result = registry.execute_tool("calculator", expression)
    print(f"结果: {result}\n")
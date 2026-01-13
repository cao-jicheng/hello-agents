import sys
sys.path.append("..")
from tools import global_tool_registry

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
    result = global_tool_registry.execute_tool("calculator", expression)
    print(f"结果: {result}\n")
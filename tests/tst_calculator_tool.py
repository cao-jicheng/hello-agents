import sys
sys.path.append("..")
from core import ToolRegistry
from tools import calculator

registry = ToolRegistry()
registry.register_function(
    name="calculator",
    description="简单的数学计算器，支持基本四则混合运算和常见函数",
    func=calculator
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
from core import OpenAICompatibleLLM
from agents import SimpleAgent
from tools import ToolRegistry, RAGTool

rag_tool = RAGTool(collection_name="test_collection", rag_namespace="my_test")
registry = ToolRegistry()
registry.register_tool(rag_tool)
client = SimpleAgent(name="知识助手", llm=OpenAICompatibleLLM())
client.tool_registry = registry

result1 = rag_tool.add_text("Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。Python的设计哲学强调代码的可读性和简洁的语法。")
print(f"知识1: {result1}")

result2 = rag_tool.add_text("机器学习是人工智能的一个分支，通过算法让计算机从数据中学习模式。主要包括监督学习、无监督学习和强化学习三种类型。")      
print(f"知识2: {result2}")

result3 = rag_tool.add_text("RAG（检索增强生成）是一种结合信息检索和文本生成的AI技术。它通过检索相关知识来增强大语言模型的生成能力。")
print(f"知识3: {result3}")

print("\n=== 搜索知识 ===")
result = rag_tool.search("Python编程语言的历史")
print(result)

print("\n=== 知识库统计 ===")
result = rag_tool.run({"action": "stats"})
print(result)

print("\n=== 清空知识库 ===")
result = rag_tool.clear_all_namespaces()
print(result)
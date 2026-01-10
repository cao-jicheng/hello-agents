import sys
sys.path.append("..")
from core import OpenAICompatibleLLM
from agents import ReActAgent

client = ReActAgent(name="AI助手", llm=OpenAICompatibleLLM())


import sys
sys.path.append("..")
from core import OpenAICompatibleLLM
from agents import SimpleAgent
from dotenv import load_dotenv

load_dotenv()

client = SimpleAgent(name="AI助手", llm=OpenAICompatibleLLM())

while True:
    try:
        text = input("我是您的AI助手，需要我做什么呢?\n")
        if "bye" in text or "exit" in text:
            print("👋 再见，期待下次为您服务！")
            break
        for chunk in client.stream_run(text):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("👋 再见，期待下次为您服务！")
        break
    except Exception as e:
        print(f"⛔ 智能体{client.name}出现错误：{str(e)}")
        break
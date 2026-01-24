from core import OpenAICompatibleLLM
from agents import SimpleAgent

client = SimpleAgent(name="AI助手", llm=OpenAICompatibleLLM())

while True:
    try:
        text = input("📢\x20我是您的AI助手，需要我做什么呢?\n")
        if "bye" in text or "exit" in text or "quit" in text:
            print("👋\x20再见，期待下次为您服务！")
            break
        for chunk in client.stream_run(text):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        print("👋\x20再见，期待下次为您服务！")
        break
    except Exception as e:
        print(f"[Agent] ⛔\x20智能体'{client.name}'出现错误：{str(e)}")
        break
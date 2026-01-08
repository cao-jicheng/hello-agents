import sys
sys.path.append("..")
from typing import Optional, Iterator
from core.agent import Agent
from core.llm import OpenAICompatibleLLM
from core.config import Config
from core.message import Message


class SimpleAgent(Agent): 
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        super().__init__(name, llm, system_prompt, config)

    def run(self, input_text: str, **kwargs) -> str:
        messages = self.stack_history_message(input_text)
        response = self.llm.invoke(messages, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
        return response
        
    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        messages = self.stack_history_message(input_text)
        full_response = ""
        for chunk in self.llm.stream_invoke(messages, **kwargs):
            full_response += chunk
            yield chunk
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = SimpleAgent(name="AI助手", llm=OpenAICompatibleLLM())
    while True:
        try:
            text = input("我是您的AI助手，需要我做什么呢：")
            for chunk in client.stream_run(text):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print("👋 期待下次再见！")
            break
        except Exception as e:
            print(f"⛔ 智能体{client.name}出现错误\n{str(e)}")
            break
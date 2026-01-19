import sys
sys.path.append("..")
from typing import Optional, Iterator
from core import Agent, OpenAICompatibleLLM, Message

class SimpleAgent(Agent): 
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt)

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
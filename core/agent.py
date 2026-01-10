from abc import ABC, abstractmethod
from typing import Optional, Any, List
from .config import Config
from .message import Message
from .llm import OpenAICompatibleLLM

class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass
    
    def add_message(self, message: Message):
        self._history.append(message)
    
    def clear_history(self):
        self._history.clear()
    
    def get_history(self) -> list[Message]:
        return self._history.copy()
    
    def stack_history_message(self, input_text: str) -> List[Message]:
        messages = []  
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content}) 
        messages.append({"role": "user", "content": input_text})
        return messages

    def __str__(self) -> str:
        return f"Agent(name={self.name}, llm_provider={self.llm.provider}, llm_model={self.llm.model})"
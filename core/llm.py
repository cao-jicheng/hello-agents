import os
from openai import OpenAI
from typing import Optional, Iterator
from .config import LLMConfig

class OpenAICompatibleLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
        ):
        config = LLMConfig.from_env()
        self.model = model or config.model
        self.base_url = base_url or config.base_url
        self.api_key = api_key or config.api_key
        self.timeout = config.timeout
        if not all([self.model, self.base_url, self.api_key]):
            raise Exception("LLM模型名称、访问网址、API密钥需要显式指定或在.env文件中定义")
        self.provider = self._auto_detect_provider()
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _auto_detect_provider(self) -> str:
        if "api.siliconflow.cn" in self.base_url:
            return "SiliconFlow"
        elif "api.deepseek.com" in self.base_url:
            return "DeepSeek"
        elif "dashscope.aliyuncs.com" in self.base_url:
            return "Qwen"
        elif "localhost:11434" in self.base_url:
            return "Ollama"
        else:
            return "Unknown"

    def invoke(self, prompts: str|list, **kwargs) -> str:
        messages = [{"role": "user", "content": prompts}] if isinstance(prompts, str) else prompts
        print(f"[LLM] 🚀\x20正在调用{self.provider}:{self.model}模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] ⛔\x20调用失败：{str(e)}")
            return ""
    
    def stream_invoke(self, prompts: str|list, **kwargs) -> Iterator[str]:
        messages = [{"role": "user", "content": prompts}] if isinstance(prompts, str) else prompts
        print(f"[LLM] 🚀\x20正在调用{self.provider}:{self.model}模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            yield "\n"
        except Exception as e:
            print(f"[LLM] ⛔\x20调用失败：{str(e)}")
            yield ""

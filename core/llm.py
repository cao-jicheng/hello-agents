import os
from typing import Literal, Optional, Iterator
from openai import OpenAI


class OpenAICompatibleLLM:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not all([self.model, self.base_url, self.api_key]):
            raise Exception("模型名称、访问网址、API密钥需要显式指定或在.env文件中定义")
        self.provider = self._auto_detect_provider()
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
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

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        print(f"🤖 正在调用{self.provider}:{self.model}模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            print("✅ LLM响应成功")
            return response.choices[0].message.content
        except Exception as e:
            print("⛔ LLM调用失败")
            return str(e)
    
    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        print(f"🤖 正在调用{self.provider}:{self.model}模型")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            print("✅ LLM响应成功")
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            yield "\n"
        except Exception as e:
            print("⛔ LLM调用失败")
            yield str(e)

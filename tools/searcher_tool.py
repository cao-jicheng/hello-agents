import sys
sys.path.append("..")
import os
import json
import requests
from typing import Dict
from core import OpenAICompatibleLLM, SearchConfig

SUMMARY_PROMPT = \
"""
## 搜索结果
{search_results}

## 目标任务
对**搜索结果**中的多项内容，进行汇总提炼，输出一段话

## 注意事项
- 提炼的要素全部来源于**搜索结果**中的内容，不要捏造虚假内容，不要新增信息
- 输出内容要简洁易懂，不要有错别字，不要前后矛盾

现在开始你的任务
"""

class SearchTool:
    def __init__(self):
        self.name = "search_tool"
        self.description = "智能搜索工具，支持多个搜索源，大模型智能提炼汇总搜索信息"
        self.search_sources = []
        self._setup()

    def _setup(self):
        config = SearchConfig.from_env()
        if config.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
                self.search_sources.append("tavily")
            except ImportError:
                print("⚠️\x20\x20tavily-python库未安装")
        if config.bocha_api_key:
            self.bocha_api_key = config.bocha_api_key
            self.search_sources.append("bocha")
        self.llm = OpenAICompatibleLLM()

    def search(self, query: str, auto_summary: bool=True) -> Dict[str, str]:
        if not query.strip():
            print("⛔\x20输入的搜索内容为空")
            return {}
        if not self.search_sources:
            print("⛔\x20没有可用的搜索源，请配置API密钥")
            return {}
        print(f"🔍\x20开始网络搜索：{query}")
        search_results = ""
        for source in self.search_sources:
            try:
                if source == "tavily":
                    search_results += self._search_with_tavily(query)
                elif source == "bocha":
                    search_results += self._search_with_bocha(query)
                print(f"✅\x20{source}已完成搜索")
            except Exception as e:
                print(f"⚠️\x20\x20{source}搜索失败：{str(e)}")
                continue
        summarized_result = ""
        if auto_summary and search_results:
            print("🎯\x20AI智能提炼汇总搜索内容")
            prompt = SUMMARY_PROMPT.format(search_results=search_results)
            summarized_result = self.llm.invoke(prompt)
        return {"search_results": search_results, "summarized_result": summarized_result}

    def _search_with_tavily(self, query: str) -> str:
        response = self.tavily_client.search(query=query, max_results=3)
        result = "=== tavily搜索到的结果 ===\n"
        for i, item in enumerate(response.get("results", []), 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"{item.get('content', '')[:1000]}\n\n"
        return result

    def _search_with_bocha(self, query: str) -> str:
        url = "https://api.bocha.cn/v1/web-search"
        headers = {
            "Authorization": f"Bearer {self.bocha_api_key}",
            "Content-Type": "application/json"
        }
        payload = json.dumps({
            "query": query,
            "summary": True,
            "count": 3
        })
        response = requests.request("POST", url, headers=headers, data=payload)
        response = response.json()
        result = "=== bocha搜索到的结果 ===\n"
        for i, item in enumerate(response["data"]["webPages"]["value"], 1):
            result += f"[{i}] {item.get('name', '')}\n"
            result += f"{item.get('summary', '')[:1000]}\n\n"
        return result

_search_tool = SearchTool()

def searcher(query: str) -> str:
    result = _search_tool.search(query, auto_summary=False)
    return result["search_results"] if result else ""

def summarized_searcher(query: str) -> str:
    result = _search_tool.search(query, auto_summary=True)
    summarized_result = ""
    if result:
        print(f"🌐\x20互联网搜索结果\n {result['search_results']}")
        summarized_result = result["summarized_result"]
    return summarized_result

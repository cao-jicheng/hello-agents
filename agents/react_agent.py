import re
import sys
sys.path.append("..")
from typing import List, Tuple, Optional, Iterator
from core import Agent, OpenAICompatibleLLM
from core import Config, Message, Tool, ToolRegistry

REACT_PROMPT = \
"""你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `tool_name(tool_param)` - 输出工具名和调用参数
- `Finish(final_answer)` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名(参数)
3. 只有当你确信有足够信息回答问题时，才可以使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""

class ReActAgent(Agent): 
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        if tool_registry is None:
            self.tool_registry = ToolRegistry()
        else:
            self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else REACT_PROMPT
    
    def add_tool(self, tool):
        if hasattr(tool, "auto_expand") and tool.auto_expand:
            if hasattr(tool, "_available_tools") and tool._available_tools:
                print(f"🛠️  MCP工具'{tool.name}'")
                for mcp_tool in tool._available_tools:
                    wrapped_tool = Tool(
                        name=f"{tool.name}_{mcp_tool['name']}",
                        description=mcp_tool.get("description", ""),
                        func=lambda input_text, t=tool, tn=mcp_tool["name"]: t.run({
                            "action": "call_tool",
                            "tool_name": tn,
                            "arguments": {"input": input_text}
                        })
                    )
                    self.tool_registry.register_tool(wrapped_tool)
            else:
                self.tool_registry.register_tool(tool)
        else:
            self.tool_registry.register_tool(tool)

    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = []
        print(f"🤖 智能体'{self.name}'开始处理用户输入：{input_text}")
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n----- 第{current_step}步 -----")
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )
            print(f"💡 提示词：\n{prompt}")
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)
            if not response_text:
                break
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🧠 思考过程：{thought}")
            if not action:
                print("⚠️  警告：未能解析出有效的Action，流程终止")
                break
            if action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案：{final_answer}")
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer
            tool_name, tool_param = self._parse_action(action)
            if not tool_name or tool_param is None:
                self.current_history.append("Observation：无效的Action格式，请检查")
                continue
            print(f"🎬 下一步行动：{tool_name}('{tool_param}')")
            observation = self.tool_registry.execute_tool(tool_name, tool_param)
            print(f"👀 结果观察：{observation}")
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")
        print("⏰ 已达到最大步数，流程终止")
        final_answer = "😭 抱歉，我无法在限定步数内完成这个任务"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer
    
    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action_match = re.search(r"Action: (.*)", text)
        action = action_match.group(1).strip() if action_match else None
        return thought, action
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.match(r"(\w+)\((.*)\)", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def _parse_action_input(self, action_text: str) -> str:
        match = re.match(r"\w+\((.*)\)", action_text)
        return match.group(1) if match else "输出内容格式解析错误"

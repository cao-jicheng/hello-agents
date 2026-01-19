import sys
sys.path.append("..")
import ast
from typing import Optional, List, Dict
from core import Agent, OpenAICompatibleLLM, Message

PLANNER_PROMPT = \
"""你是一个顶级的任务规划专家，你可以将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

## 初始问题
{question}

## 工作流程
请严格按照以下格式输出你的计划：
```python
["步骤1", "步骤2", "步骤3", ...]
```

## 重要提醒
- 确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列
- 你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串
"""

EXECUTOR_PROMPT = \
"""你是一位顶级的任务执行专家，你能够严格按照给定的计划，一步步地解决问题。

## 初始问题
{question}

## 完整计划
{plan}

## 历史步骤与结果
{history}

## 当前步骤:
{current_step}

## 工作流程
你将根据**初始问题**、**完整计划**、和**历史步骤与结果**，仅针对**当前步骤**输出答案。
不要输出任何额外的解释或对话。
"""

class Planner:
    def __init__(
        self,
        llm: OpenAICompatibleLLM, 
        custom_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.prompt_template = custom_prompt if custom_prompt else PLANNER_PROMPT

    def plan(self, question: str, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(question=question)
        print(f"💡\x20Planner提示词：\n{prompt}")
        response_text = self.llm.invoke(prompt, **kwargs)
        print(f"🧮\x20AI已完成任务规划：\n{response_text}")
        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else None
        except Exception as e:
            print(f"⛔\x20解析任务规划文本出错： {str(e)}")
            return None

class Executor:
    def __init__(
        self,
        llm: OpenAICompatibleLLM, 
        custom_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.prompt_template = custom_prompt if custom_prompt else EXECUTOR_PROMPT

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        history = ""
        final_answer = ""
        for i, step in enumerate(plan, 1):
            print(f"🎬\x20正在执行步骤{i}/{len(plan)}：{step}")
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            print(f"💡\x20Executor提示词：\n{prompt}")
            response_text = self.llm.invoke(prompt, **kwargs)
            history += f"步骤{i}：{step}\n执行结果：{response_text}\n"
            final_answer = response_text
            print(f"✅\x20步骤{i} 已完成，结果：{final_answer}")
        return final_answer

class PlanAndExecuteAgent(Agent):  
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        system_prompt: Optional[str] = None,
        custom_prompt: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt)
        if custom_prompt:
            planner_prompt = custom_prompt.get("planner")
            executor_prompt = custom_prompt.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None
        self.planner = Planner(self.llm, planner_prompt)
        self.executor = Executor(self.llm, executor_prompt)
    
    def run(self, input_text: str, **kwargs) -> str:
        print(f"🤖\x20智能体'{self.name}'开始处理问题：{input_text}")
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "AI无法生成有效的行动计划，任务终止"
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            print(f"⛔\x20智能体运行出错：{final_answer}")
            return final_answer
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        print(f"🎉\x20最终答案：{final_answer}")
        return final_answer
from typing import Optional, List, Dict, Any
from core import Agent, OpenAICompatibleLLM, Message

REFLECTION_PROMPT = {
    "initial": """
## 初始问题
{question}

请针对以上问题，输出一个完整准确的回答
""",
    "reflect": """
## 初始问题
{question}

## 当前回答
{answer}

## 工作流程
请仔细审查**当前回答**的内容，并找出可能的问题或改进空间：
- 如果存在不足之处，指出来并提供具体的改进建议
- 如果回答已经很好，请直接回答"无需改进"
""",
    "refine": """
## 初始问题
{question}

## 当前回答
{answer}

## 反馈意见
{feedback}

请根据以上信息，输出一个改进后的回答
"""
}

class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        self.records.append({"type": record_type, "content": content})
        print(f"[Agent] 记忆已更新，新增一条'{record_type}'记录")

    def get_trajectory(self) -> str:
        trajectory = ""
        for record in self.records:
            if record["type"] == "execution":
                trajectory += f"--- 上一轮执行结果 ---\n{record['content']}\n"
            elif record["type"] == "reflection":
                trajectory += f"--- 评审员反馈 ---\n{record['content']}\n"
        return trajectory.strip()

    def get_last_execution(self) -> str:
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

class ReflectionAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: OpenAICompatibleLLM,
        system_prompt: Optional[str] = None,
        custom_prompt: Optional[Dict[str, str]] = None,
        max_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt)
        self.max_iterations = max_iterations
        self.prompt_template = custom_prompt if custom_prompt else REFLECTION_PROMPT
    
    def run(self, input_text: str, **kwargs) -> str:
        print(f"🤖\x20智能体'{self.name}'开始处理问题：{input_text}")
        self.memory = Memory()
        initial_prompt = self.prompt_template["initial"].format(question=input_text)
        print(f"[Agent] 初始提示词：\n{initial_prompt}")
        initial_result = self.llm.invoke(initial_prompt, **kwargs)
        self.memory.add_record("execution", initial_result)
        for i in range(self.max_iterations):
            print(f"\n----- 第 {i+1}/{self.max_iterations} 轮迭代 -----")
            last_result = self.memory.get_last_execution()
            reflect_prompt = self.prompt_template["reflect"].format(
                question=input_text,
                answer=last_result
            )
            print(f"[Agent] 反思提示词：\n{reflect_prompt}")
            feedback = self.llm.invoke(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)
            if "无需改进" in feedback:
                print("[Agent] AI认为结果已无需改进，任务完成")
                break
            refine_prompt = self.prompt_template["refine"].format(
                question=input_text,
                answer=last_result,
                feedback=feedback
            )
            print(f"[Agent] 改进提示词：\n{refine_prompt}")
            refined_result = self.llm.invoke(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined_result)
        final_result = self.memory.get_last_execution()
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        print(f"\n🎉\x20最终答案：{final_result}")
        return final_result
    

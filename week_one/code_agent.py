from typing import Dict, List
from agents.agents import Agent, OllamaAIAgent, OpenAIAgent
from utils.utils import create_message


class CodeAnalyzer:
    def __init__(self, code: str):
        self.message = self.create_message(code)

    def get_system_prompt(self) -> str:
        return (
            "You are a coding tutor that explains what a block of code does "
            "and provides a better alternative code block to achieve the same "
            "result if any."
        )

    # def get_user_prompt(self, code: str) -> str:
    #     return "Please explain what this code does and why:\n" + code

    def create_message(self, code: str) -> List[Dict[str, str]]:
        return create_message(
            system_prompt=self.get_system_prompt(),
            user_prompt=code
        )

    def analyze_code(self, model: str, agent: Agent) -> str:
        return agent.respond_with(
            model=model,
            messages=self.message,
        )

    def analyze_with_llama(self, model: str) -> str:
        agent = OllamaAIAgent()
        return self.analyze_code(
            model=model,
            agent=agent,
        )

    def analyze_with_openai(self, model: str) -> str:
        agent = OpenAIAgent()
        return self.analyze_code(
            model=model,
            agent=agent,
        )

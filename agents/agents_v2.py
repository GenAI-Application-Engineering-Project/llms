from abc import ABC, abstractmethod

import anthropic
from openai import OpenAI


class Agent(ABC):

    @abstractmethod
    def respond(
        self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
    ) -> str:
        pass

    @abstractmethod
    def respond_with_stream(
        self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
    ) -> str:
        pass


class OpenAIAgent(Agent):
    def __init__(self):
        self.openai = OpenAI()

    def respond(
        self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completions = self.openai.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completions.choices[0].message.content

    def respond_with_stream(
        self, system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini"
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        stream = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        result = ""
        for chunk in stream:
            result += chunk.choices[0].delta.content or ""
            yield result


class ClaudeAgent(Agent):
    def __init__(self):
        self.agent = anthropic.Anthropic()

    def respond(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-3-haiku-20240307"
    ) -> str:
        message = self.agent.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def respond_with_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "claude-3-haiku-20240307"
    ):
        response = self.agent.messages.stream(
            model=model,
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        result = ""
        with response as stream:
            for text in stream.text_stream:
                result += text or ""
                yield result

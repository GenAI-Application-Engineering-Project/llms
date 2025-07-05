from abc import ABC, abstractmethod
import json
from typing import Dict, List, Optional

import ollama
from openai import OpenAI


class Agent(ABC):
    """
    Abstract base class for LLM agents capable of chatting using a specific
    model.
    """

    def __init__(
        self, model: str = "", messages: Optional[List[Dict[str, str]]] = None
    ):
        """
        Initialize the agent with an optional model name and message history.
        """
        self.model = model
        self.messages = messages if messages is not None else []

    def respond(self) -> str:
        """
        Generate a response using the internal model and message history.
        """
        return self.respond_with(self.model, self.messages)

    @abstractmethod
    def respond_with(self, model: str, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the provided model and messages.
        """
        pass

    @abstractmethod
    def respond_with_json(
        self, model: str, messages: List[Dict[str, str]]
    ):
        """
        Generate a response using the provided model and messages.
        """
        pass


class OpenAIAgent(Agent):
    def __init__(
        self, model: str = "gpt-4o-mini",
        messages: Optional[List[Dict[str, str]]] = None
    ):
        super().__init__(model, messages)

    def respond_with(self, model: str, messages: List[Dict[str, str]]) -> str:
        openai = OpenAI()
        response = openai.chat.completions.create(
            model=model,
            messages=messages
        )
        response = response.choices[0].message.content
        return response

    def respond_with_json(
        self, model: str, messages: List[Dict[str, str]]
    ):
        openai = OpenAI()
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        result = response.choices[0].message.content
        return json.loads(result)


class OllamaAIAgent(Agent):
    def __init__(
        self, model: str = "llama3.2",
        messages: Optional[List[Dict[str, str]]] = None
    ):
        super().__init__(model, messages)

    def respond_with(self, model: str, messages: List[Dict[str, str]]) -> str:
        response = ollama.chat(
            model=model,
            messages=messages,
        )
        return response['message']['content']

    def respond_with_json(
        self, model: str, messages: List[Dict[str, str]]
    ) -> str:
        response = ollama.chat(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response['message']['content']

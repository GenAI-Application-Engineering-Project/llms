from abc import ABC, abstractmethod
from dataclasses import dataclass

import ollama
from openai import OpenAI


@dataclass
class Website:
    url: str
    title: str
    body: str


class Summarizer(ABC):
    def __init__(self, website: Website):
        self.website = website

    @abstractmethod
    def summarize(self) -> str:
        pass

    def get_system_prompt(self) -> str:
        return (
            "You are an assistant that analyzes the contents of a website "
            "and provides a short summary, ignoring text that might be "
            "navigation related. Respond in markdown."
        )

    def get_user_prompt(self) -> str:
        return (
            f"You are looking at a website titled {self.website.title}"
            "\nThe contents of this website is as follows;"
            "please provide a short summary of this website in markdown. "
            "If it includes news or announcements, then summarize these too."
            f"\n\n {self.website.body}"
        )

    def create_message(self):
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class OllamaSummarizer(Summarizer):
    def summarize(self) -> str:
        response = ollama.chat(
            model="llama3.2",
            messages=self.create_message()
        )
        return response['message']['content']


class OpenAISummarizer(Summarizer):
    def summarize(self) -> str:
        openai = OpenAI()
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.create_message()
        )
        return response.choices[0].message.content

from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import OpenAI

from scraper.scraper import Scraper


@dataclass
class Website:
    url: str
    title: str
    body: str


def get_api_key():
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print(
            "No API key was found - please head over to the troubleshooting \
                notebook in this folder to identify & fix!"
        )
    elif not api_key.startswith("sk-proj-"):
        print(
            "An API key was found, but it doesn't start sk-proj-; please \
                check you're using the right key - see troubleshooting \
                    notebook"
        )
    elif api_key.strip() != api_key:
        print(
            "An API key was found, but it looks like it might have space or \
                tab characters at the start or end - please remove them - see \
                    troubleshooting notebook"
        )
    else:
        print("API key found and looks good so far!")


def get_system_prompt() -> str:
    return "You are an assistant that analyzes the contents of a website \
        and provides a short summary, ignoring text that might be navigation \
            related. Respond in markdown."


def get_user_prompt(website: Website) -> str:
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
        please provide a short summary of this website in markdown. \
            If it includes news or announcements, then summarize these too.\
                \n\n"
    user_prompt += website.body
    return user_prompt


def create_message(website):
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(website)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def summarize_website(scraper: Scraper) -> str:
    website = Website(
        url=scraper.get_url(),
        title=scraper.get_title(),
        body=scraper.get_body()
    )

    openai = OpenAI()
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=create_message(website)
    )

    return response.choices[0].message.content

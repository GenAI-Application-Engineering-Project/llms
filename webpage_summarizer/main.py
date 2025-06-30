import os
from dotenv import load_dotenv
from scraper.scraper import SoupScraper
from webpage_summarizer.summarizer import (
    OllamaSummarizer, OpenAISummarizer, Summarizer, Website
)


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


def create_summary(summarizer: Summarizer, model: str) -> None:
    summary = summarizer.summarize()
    print(f"Summary by {model}")
    print(summary)
    print("\n\n")


def main():
    get_api_key()

    url = "https://edwarddonner.com"
    scraper = SoupScraper(url)
    website = Website(
        url=scraper.get_url(),
        title=scraper.get_title(),
        body=scraper.get_body()
    )

    ollama_summarizer = OllamaSummarizer(website)
    create_summary(ollama_summarizer, "Ollama")

    openai_summarizer = OpenAISummarizer(website)
    create_summary(openai_summarizer, "OpenAI")


if __name__ == "__main__":
    main()

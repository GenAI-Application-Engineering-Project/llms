from scraper.scraper import SoupScraper
from utils.utils import load_env_variables
from webpage_summarizer.summarizer import (
    OllamaSummarizer, OpenAISummarizer, Summarizer, Website
)


def create_summary(summarizer: Summarizer, model: str) -> None:
    summary = summarizer.summarize()
    print(f"Summary by {model}")
    print(summary)
    print("\n\n")


def main():
    load_env_variables()

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

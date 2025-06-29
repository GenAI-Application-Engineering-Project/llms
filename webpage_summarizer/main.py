from scraper.scraper import SoupScraper
from webpage_summarizer.website import get_api_key, summarize_website


def main():
    get_api_key()
    # url = "https://edwarddonner.com"
    url = "https://enviromedica.com"
    scraper = SoupScraper(url)
    summary = summarize_website(scraper)
    print(summary)


if __name__ == "__main__":
    main()

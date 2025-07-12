from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import List

from bs4 import BeautifulSoup
import requests

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}


@dataclass
class Website:
    url: str
    title: str
    body: str
    links: str


class Scraper(ABC):
    @abstractmethod
    def get_url(self):
        """Return the URL of the page."""
        pass

    @abstractmethod
    def get_title(self) -> str:
        """Extract the page title."""
        pass

    @abstractmethod
    def get_body(self, exclude_tags: list = None) -> str:
        """Extract the main body text of the page."""
        pass

    def get_contents(self) -> str:
        return (
            f"Webpage Title:\n{self.get_title()}"
            f"\nWebpage Contents:\n{self.get_body()}\n\n"
        )


class SoupScraper(Scraper):
    def __init__(self, url: str, headers: dict = None):
        """
        Initialize the scraper with a URL and optional headers.
        Fetch the content and parse it with BeautifulSoup.
        """
        super().__init__()
        self.url = url
        self.headers = headers or DEFAULT_HEADERS
        try:
            response = requests.get(self.url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch URL {self.url}: {e}")
            raise

        self.soup = BeautifulSoup(response.content, 'html.parser')

    def get_url(self):
        """Return the URL of the page."""
        return self.url

    def get_title(self):
        """Extract the page title, or fallback if not available."""
        return self.soup.title.string if self.soup.title else "No title found"

    def get_body(self, exclude_tags=None) -> str:
        """
        Extract the body text of the page, excluding certain HTML elements.

        Args:
            exclude_tags (list): List of tag names to remove. Defaults to
            ["script", "style", "img", "input"].

        Returns:
            str: Cleaned body text.
        """
        if not self.soup.body:
            return "No body found"

        exclude_tags = exclude_tags or ["script", "style", "img", "input"]
        for tag in self.soup.body(exclude_tags):
            tag.decompose()

        return self.soup.body.get_text(separator="\n", strip=True)

    def get_links(self) -> List[str]:
        links = [link.get('href') for link in self.soup.find_all('a')]
        return [link for link in links if link]

    def to_dict(self) -> dict:
        """Return a dictionary representation of the scraped content."""
        return {
            "url": self.get_url(),
            "title": self.get_title(),
            "body": self.get_body(),
            "links": self.get_links()
        }

    def get_website_content(self) -> Website:
        return Website(
            url=self.get_url(),
            title=self.get_title(),
            body=self.get_body(),
            links=self.get_links()
        )

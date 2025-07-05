from typing import Dict, List

from scraper.scraper import Website
from utils.utils import create_message


class MessageCreator:
    def get_link_system_prompt(self) -> str:
        """
        Returns a system prompt for identifying and classifying relevant
        company-related links from a webpage for inclusion in a brochure.
        """
        system_prompt = (
            "You are provided with a list of links found on a webpage. You "
            "are able to decide which of the links would be most relevant to "
            "include in a brochure about the company, such as links to an "
            "About page, or a Company page, or Careers/Jobs pages.\n"
            "You should respond in JSON as in this example:"
        )
        example = """
        {
            "links": [
                {
                    "type": "about page",
                    "url": "https://full.url/goes/here/about"
                },
                {
                    "type": "careers page",
                    "url": "https://another.full.url/careers"
                }
            ]
        }
        """
        return f"{system_prompt}\n{example}"

    def get_links_user_prompt(self, website: Website) -> str:
        """
        Generates a user prompt for identifying relevant links from a website
        for inclusion in a company brochure.

        Args:
            website (Website): A Website object containing the base URL and a
            list of links found on the site.

        Returns:
            str: A formatted prompt string that includes guidance and the raw
            list of links for evaluation.
        """
        return (
            f"Here is the list of links on the website of {website.url} - "
            "please decide which of these are relevant web links for a "
            "brochure about the company, respond with the full https URL in "
            "JSON format. Do not include Terms of Service, Privacy, email "
            f"links.\nLinks (some might be relative links):\n\n{website.links}"
        )

    def get_brochure_user_prompt(self, company_name: str, details: str) -> str:
        user_prompt = (
            f"You are looking at a company called: {company_name}\n"
            "Here are the contents of its landing page and other relevant "
            "pages; use this information to build a short brochure of the "
            f"company in markdown.\n{details}"
        )
        return user_prompt[:5_000]  # Truncate if more than 5,000 characters

    def get_brochure_system_prompt(self) -> str:
        return (
            "You are an assistant that analyzes the contents of several "
            "relevant pages from a company website and creates a short "
            "brochure about the company for prospective customers, investors "
            "and recruits. Respond in markdown. Include details of company "
            "culture, customers and careers/jobs if you have the information."
        )

    def create_links_message(self, website: Website) -> List[Dict[str, str]]:
        system_prompt = self.get_link_system_prompt()
        user_prompt = self.get_links_user_prompt(website)
        message = create_message(system_prompt, user_prompt)
        return message

    def create_brochure_message(
        self, company_name: str, details: str
    ) -> List[Dict[str, str]]:
        system_prompt = self.get_brochure_system_prompt()
        user_prompt = self.get_brochure_user_prompt(company_name, details)
        return create_message(system_prompt, user_prompt)

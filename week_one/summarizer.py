from scraper.scraper import Website


class Summarizer:
    def get_system_prompt(self) -> str:
        return (
            "You are an assistant that analyzes the contents of a website "
            "and provides a short summary, ignoring text that might be "
            "navigation related. Respond in markdown."
        )

    def get_user_prompt(self, website: Website) -> str:
        return (
            f"You are looking at a website titled {website.title}"
            "\nThe contents of this website is as follows;"
            "please provide a short summary of this website in markdown. "
            "If it includes news or announcements, then summarize these too."
            f"\n\n {website.body}"
        )

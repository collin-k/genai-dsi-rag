"""Clean scraped HTML and write structured text files to the knowledge base."""

from config import RAW_DIR
from pathlib import Path
import logging
from bs4 import BeautifulSoup

def load_raw_html(path: Path) -> str:
    """Load raw HTML from a file."""

def extract_main_content(soup: BeautifulSoup) -> str:
    """Extract the main content of the page that drops noise (nav, footer, scripts, etc.)."""

def extract_text(content: str) -> str:
    """Extract the text of the page."""

def normalize_text(text: str) -> str:
    """Normalize the text of the page."""

def save_cleaned_text(text: str, filename: str, cleaned_dir: Path) -> None:
    """Save the cleaned text to a file and save to 'data/cleaned/."""

def main() -> None:
    """CLI: python src/preprocess.py"""
    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing of raw HTML files.")
    for path in RAW_DIR.glob("*.html"):
        with open(path, "r") as file:
            html = file.read()
            print(html)
            # TODO: Implement the preprocessing steps
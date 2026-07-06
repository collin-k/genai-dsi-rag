"""Scrape the DSI webpage and sublinks."""

from config import URL, RAW_DIR
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import logging

def fetch_page(url: str, timeout: int = 15) -> str:
    """GET a URL and return raw HTML. Raises on HTTP errors."""

def url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename, e.g. curriculum.html."""

def save_raw_html(html: str, url: str, raw_dir: Path) -> Path:
    """Write HTML to data/raw/ and return the file path."""

def extract_links(html: str, base_url: str) -> list[str]:
    """Return absolute hrefs found on the page."""

def is_valid_program_link(url: str, seed_url: str) -> bool:
    """
    Filter links to relevant same-site program pages.
    Exclude: external sites, PDFs, mailto, login, duplicate nav links, etc.
    """

def collect_urls(start_url: str, max_pages: int = 50) -> list[str]:
    """
    BFS crawl from the seed URL.
    Visit main page + program-related sublinks (curriculum, admissions, etc.).
    """

def main() -> None:
    """CLI: python src/scrape.py"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting scrape of {URL} with max_pages={50}")
    collected_urls = collect_urls(URL, max_pages=50)
    print(f"Collected {len(collected_urls)} URLs.")
    logger.info(f"Collected {len(collected_urls)} URLs.")
    for link in collected_urls:
        if is_valid_program_link(link, URL):
            content = fetch_page(link, RAW_DIR)
            save_raw_html(content, link, RAW_DIR)
            print(f"Saved {link} to {RAW_DIR}")
            logger.info(f"Saved {link} to {RAW_DIR}")
        else:
            continue
        
"""Scrape the DSI webpage and sublinks."""

import logging
import re
from collections import deque
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import (
    MAX_SCRAPE_PAGES,
    PROGRAM_PATH_PREFIX,
    RAW_DIR,
    RELATED_PROGRAM_URLS,
    REQUEST_TIMEOUT,
    URL,
    USER_AGENT,
    NON_HTML_EXTENSIONS,
    EXCLUDED_PATH_FRAGMENTS,
)


def fetch_page(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    GET a URL and return raw HTML.

    Parameters
    ----------
    url : str
        Page URL to fetch.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Raw HTML response body.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-success status code.
    requests.RequestException
        If the request fails for network-related reasons.
    """
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def normalize_url(url: str) -> str:
    """
    Normalize a URL for deduplication and filesystem-safe storage.

    Parameters
    ----------
    url : str
        Absolute or relative URL.

    Returns
    -------
    str
        Canonical URL without fragments, with a trailing slash.
    """
    cleaned_url = unquote(url).strip().split("#", maxsplit=1)[0].strip()
    parsed = urlparse(cleaned_url)
    path = parsed.path.strip()
    path = re.sub(r"\s+/?$", "/", path)
    if not path.endswith("/"):
        path = f"{path}/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def url_to_filename(url: str) -> str:
    """
    Convert a URL to a safe filename.

    Parameters
    ----------
    url : str
        Canonical page URL.

    Returns
    -------
    str
        Filename such as ``education__masters-programs__ms-in-applied-data-science__faqs.html``.
    """
    parsed = urlparse(normalize_url(url))
    path = parsed.path.strip("/")
    if not path:
        return "index.html"
    return f"{path.replace('/', '__')}.html"


def save_raw_html(html: str, url: str, raw_dir: Path) -> Path:
    """
    Write HTML to ``data/raw/`` and return the file path.

    Parameters
    ----------
    html : str
        Raw HTML content.
    url : str
        Source page URL.
    raw_dir : Path
        Directory for scraped HTML files.

    Returns
    -------
    Path
        Path to the saved HTML file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / url_to_filename(url)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def extract_links(html: str, base_url: str) -> list[str]:
    """
    Return absolute, normalized hrefs found on the page.

    Parameters
    ----------
    html : str
        Raw HTML content.
    base_url : str
        URL used to resolve relative links.

    Returns
    -------
    list[str]
        Sorted list of unique absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        absolute_url = normalize_url(urljoin(base_url, href))
        links.add(absolute_url)

    return sorted(links)


def is_valid_program_link(url: str, seed_url: str) -> bool:
    """
    Filter links to relevant same-site MS-ADS program pages.

    Includes pages under the MS-ADS path prefix and a small set of related
    sibling program pages linked from the hub. Excludes external sites, asset
    files, and other DSI degree programs.

    Parameters
    ----------
    url : str
        Candidate URL to validate.
    seed_url : str
        Crawl seed URL. Reserved for future seed-relative rules.

    Returns
    -------
    bool
        True if the URL should be crawled and saved.
    """
    _ = seed_url
    normalized_url = normalize_url(url)
    parsed = urlparse(normalized_url)

    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc != "datascience.uchicago.edu":
        return False
    if parsed.path.lower().endswith(NON_HTML_EXTENSIONS):
        return False
    if any(fragment in parsed.path for fragment in EXCLUDED_PATH_FRAGMENTS):
        return False
    if "%20" in parsed.path:
        return False

    if normalized_url.startswith(PROGRAM_PATH_PREFIX):
        return True

    return normalized_url in {normalize_url(related_url) for related_url in RELATED_PROGRAM_URLS}


def collect_urls(start_url: str, max_pages: int = MAX_SCRAPE_PAGES) -> list[str]:
    """
    Breadth-first crawl from the seed URL.

    Visits the MS-ADS hub and program-related subpages such as admissions,
    curriculum, tuition, and FAQs.

    Parameters
    ----------
    start_url : str
        Seed URL for the crawl.
    max_pages : int
        Maximum number of pages to collect.

    Returns
    -------
    list[str]
        Sorted list of discovered program URLs.
    """
    seed = normalize_url(start_url)
    queue = deque([seed])
    visited = set()
    collected = []

    while queue and len(collected) < max_pages:
        current_url = queue.popleft()
        if current_url in visited:
            continue

        visited.add(current_url)
        if not is_valid_program_link(current_url, seed):
            continue

        try:
            html = fetch_page(current_url)
        except requests.RequestException as error:
            logging.getLogger(__name__).warning("Skipping %s: %s", current_url, error)
            continue

        collected.append(current_url)

        for link in extract_links(html, current_url):
            if link not in visited and is_valid_program_link(link, seed):
                queue.append(link)

    return sorted(collected)


def main() -> None:
    """CLI entry point: ``python src/scrape.py``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting scrape of %s (max_pages=%s)", URL, MAX_SCRAPE_PAGES)
    collected_urls = collect_urls(URL, max_pages=MAX_SCRAPE_PAGES)
    logger.info("Collected %s URLs", len(collected_urls))

    for link in collected_urls:
        html = fetch_page(link)
        _ = save_raw_html(html, link, RAW_DIR)

    print(f"Scraped {len(collected_urls)} pages into {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()

"""Clean scraped HTML and write structured text files to the knowledge base."""

from bs4 import BeautifulSoup, Tag
import re
from pathlib import Path

from config import (
    RAW_DIR, 
    CLEANED_DIR,
    NON_HTML_EXTENSIONS,
    EXCLUDED_PATH_FRAGMENTS,
    MAIN_CONTENT_SELECTORS,
    NOISE_SELECTORS,
    BLOCK_TAGS,
)

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def extract_content(soup: BeautifulSoup) -> Tag:
    """
    Return the main content region with non-content elements removed.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML document.

    Returns
    -------
    Tag
        Main content subtree, or the document body when no main
        content region is found.
    """
    content = None
    for selector in MAIN_CONTENT_SELECTORS:
        content = soup.select_one(selector)
        if content is not None:
            break

    if content is None:
        content = soup.body or soup

    for selector in NOISE_SELECTORS:
        for element in content.select(selector):
            element.decompose()

    return content

def extract_text(content: Tag) -> str:
    """
    Extract section-aware text with the heading hierarchy preserved.

    Parameters
    ----------
    content : Tag
        Main content subtree.

    Returns
    -------
    str
        Markdown-style text with headings and body paragraphs.
    """
    parts = []

    for element in content.find_all(BLOCK_TAGS):
        text = element.get_text(" ", strip=True)
        if not text:
            continue

        if element.name in HEADING_TAGS:
            heading_prefix = "#" * int(element.name[1])
            parts.append(f"{heading_prefix} {text}")
        else:
            parts.append(text)

    return "\n\n".join(parts)

def normalize_text(text: str) -> str:
    '''Collapse redundant whitespace and standardize paragraph breaks.'''
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_html(html: str, output_dir: Path) -> str:
    '''
    Clean the HTML
    '''
    soup = BeautifulSoup(html, "html.parser")
    content = extract_content(soup)
    text = extract_text(content)
    clean_text = normalize_text(text)
    
    return clean_text


def recurse_html_files(dir: Path, output_dir: Path):
    for file in dir.glob("*.html"):
        
        if (file.name.endswith(NON_HTML_EXTENSIONS) 
        or any(fragment in file.name for fragment in EXCLUDED_PATH_FRAGMENTS)):
            continue

        text = clean_html(file.read_text(), output_dir)
        
        with open(output_dir.joinpath(file.name.replace(".html", ".txt")), "w") as f:
            f.write(text)


def preprocess():
    recurse_html_files(RAW_DIR, CLEANED_DIR)
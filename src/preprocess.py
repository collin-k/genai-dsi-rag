"""Clean scraped HTML and write structured text files to the knowledge base."""

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from config import (
    BLOCK_TAGS,
    CLEANED_DIR,
    EXCLUDED_PATH_FRAGMENTS,
    MAIN_CONTENT_SELECTORS,
    NOISE_SELECTORS,
    NON_HTML_EXTENSIONS,
    RAW_DIR,
)

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
CORE_COURSE_SECTION_TITLE = "Core Courses"
CAPSTONE_TITLE_MARKERS = ("capstone",)


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

    promote_accordion_titles(content)
    inject_core_course_summaries(content)
    return content


def promote_accordion_titles(content: Tag) -> None:
    """
    Convert accordion titles into heading tags for section-aware extraction.

    Top-level accordion group titles become ``h2`` elements. Nested course or
    FAQ titles become ``h3`` elements.

    Parameters
    ----------
    content : Tag
        Main content subtree to update in place.
    """
    for title in list(content.select("a.accordion-title")):
        text = title.get_text(" ", strip=True)
        if not text:
            title.decompose()
            continue

        parent_item = title.find_parent("li", class_=True)
        classes = parent_item.get("class", []) if parent_item else []
        level = 3 if "accordion__item" in classes else 2

        heading = content.new_tag(f"h{level}")
        heading.string = text
        title.replace_with(heading)


def inject_core_course_summaries(content: Tag) -> None:
    """
    Insert an explicit core-course list under each Core Courses heading.

    Parameters
    ----------
    content : Tag
        Main content subtree containing promoted accordion headings.
    """
    for heading in list(content.find_all("h2")):
        if heading.get_text(" ", strip=True) != CORE_COURSE_SECTION_TITLE:
            continue

        course_names = []
        for sibling in heading.find_all_next(["h2", "h3"]):
            if sibling.name == "h2":
                break
            name = sibling.get_text(" ", strip=True)
            if name and not _is_capstone_title(name):
                course_names.append(name)

        if not course_names:
            continue

        summary = content.new_tag("p")
        summary.string = (
            "The core courses in the MS in Applied Data Science program are: "
            + "; ".join(course_names)
            + "."
        )
        heading.insert_after(summary)


def _is_capstone_title(title: str) -> bool:
    """
    Return whether a course title refers to the Capstone project.

    Parameters
    ----------
    title : str
        Course or accordion title.

    Returns
    -------
    bool
        True when the title is a Capstone listing.
    """
    lowered = title.lower()
    return any(marker in lowered for marker in CAPSTONE_TITLE_MARKERS)


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
    block_tags = frozenset(BLOCK_TAGS)
    parts = []

    for element in content.find_all(True):
        if element.name not in block_tags:
            continue
        if _is_accordion_list_item(element):
            continue
        if _has_block_ancestor(element, block_tags):
            continue

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
    """
    Collapse redundant whitespace and standardize paragraph breaks.

    Parameters
    ----------
    text : str
        Extracted page text.

    Returns
    -------
    str
        Normalized text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def enrich_schedule_core_courses(text: str) -> str:
    """
    Add an explicit Core Courses section parsed from schedule blobs.

    Course Progressions pages often embed labels like ``Core Machine Learning I``
    inside a single paragraph. This helper extracts those titles into a
    dedicated markdown section so retrieval can answer curriculum questions.

    Parameters
    ----------
    text : str
        Cleaned page text.

    Returns
    -------
    str
        Text with a synthetic Core Courses section when names are found and
        no dedicated Core Courses section already exists.
    """
    if re.search(r"^##\s+Core Courses\s*$", text, flags=re.MULTILINE):
        return text

    names = []
    for match in re.finditer(r"\bCore\s+([A-Z][^.\n]*?)(?:\s+Letter Grade|\s+Pass/Fail)", text):
        name = match.group(1).strip(" :-")
        name = re.sub(r"\s+", " ", name)
        if name and not _is_capstone_title(name) and name not in names:
            names.append(name)

    if len(names) < 3:
        return text

    summary = (
        "## Core Courses\n\n"
        "The core courses in the MS in Applied Data Science program are: "
        + "; ".join(names)
        + ".\n\n"
    )
    return summary + text


def clean_html(html: str) -> str:
    """
    Convert raw HTML into normalized, section-aware plain text.

    Parameters
    ----------
    html : str
        Raw HTML page content.

    Returns
    -------
    str
        Cleaned text ready for chunking.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = extract_content(soup)
    text = extract_text(content)
    text = enrich_schedule_core_courses(text)
    return normalize_text(text)


def recurse_html_files(raw_dir: Path, output_dir: Path) -> int:
    """
    Clean every scraped HTML file into ``data/cleaned/``.

    Parameters
    ----------
    raw_dir : Path
        Directory containing scraped HTML files.
    output_dir : Path
        Destination directory for cleaned text files.

    Returns
    -------
    int
        Number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for file in sorted(raw_dir.glob("*.html")):
        if file.name.endswith(NON_HTML_EXTENSIONS):
            continue
        if any(fragment.strip("/") in file.name for fragment in EXCLUDED_PATH_FRAGMENTS):
            continue

        text = clean_html(file.read_text(encoding="utf-8"))
        output_path = output_dir / file.name.replace(".html", ".txt")
        output_path.write_text(text, encoding="utf-8")
        written += 1

    return written


def preprocess() -> None:
    """CLI entry point used by ``python src/preprocess.py``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    count = recurse_html_files(RAW_DIR, CLEANED_DIR)
    logger.info("Wrote %s cleaned files to %s", count, CLEANED_DIR)


def _is_accordion_list_item(element: Tag) -> bool:
    """
    Return whether an element is an accordion wrapper list item.

    Parameters
    ----------
    element : Tag
        Candidate HTML element.

    Returns
    -------
    bool
        True for accordion item containers whose text is represented by
        promoted headings and nested paragraphs instead.
    """
    if element.name != "li":
        return False

    classes = element.get("class", [])
    return "accordion-item" in classes or "accordion__item" in classes


def _has_block_ancestor(element: Tag, block_tags: frozenset) -> bool:
    """
    Check whether an element is nested inside another extractable block.

    Parameters
    ----------
    element : Tag
        Candidate text block.
    block_tags : frozenset
        Block tag names treated as extractable units.

    Returns
    -------
    bool
        True when a parent block should own the text instead.
    """
    for parent in element.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name in HEADING_TAGS:
            return True
        if parent.name in block_tags:
            classes = parent.get("class", [])
            if parent.name == "li" and (
                "accordion-item" in classes or "accordion__item" in classes
            ):
                continue
            return True
    return False


if __name__ == "__main__":
    preprocess()

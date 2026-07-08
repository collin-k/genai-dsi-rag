"""Shared configuration: URLs, file paths, chunk sizes, and model settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
PROGRAM_PATH_PREFIX = (
    "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science"
)
RELATED_PROGRAM_URLS = (
    "https://datascience.uchicago.edu/education/masters-programs/in-person-program/",
    "https://datascience.uchicago.edu/education/masters-programs/online-program/",
)
MAX_SCRAPE_PAGES = 50
REQUEST_TIMEOUT = 15
USER_AGENT = "UChicago-MSADS-RAG-Bot/1.0 (educational project)"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"

NON_HTML_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".zip",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)
EXCLUDED_PATH_FRAGMENTS = (
    "/computational-analysis-and-public-policy/",
    "/phd-in-data-science/",
    "/undergrad-major/",
    "/data-science-clinic/",
    "/summer-research-programs/",
)
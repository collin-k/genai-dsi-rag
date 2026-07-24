"""Shared configuration: URLs, file paths, chunk sizes, and model settings."""

from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

URL = "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/"
BASE_URL = f"{urlparse(URL).scheme}://{urlparse(URL).netloc}"
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

BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dd", "blockquote")
NOISE_SELECTORS = ("script", "style", "noscript", ".gridder-list", ".button--read-more", ".button--read-less")
MAIN_CONTENT_SELECTORS = (".main-content", "main.site-content", "main")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "text-embedding-3-small"
TIKTOKEN_ENCODING = "cl100k_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

RETRIEVER_TOP_K = 4
HYBRID_WEIGHTS = (0.5, 0.5)
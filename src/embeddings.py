"""Chunk cleaned documents and build the FAISS embedding store for program Q&A."""

import os
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    BASE_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CLEANED_DIR,
    EMBEDDING_MODEL,
    TIKTOKEN_ENCODING,
    VECTOR_STORE_DIR,
)

WORD_PATTERN = re.compile(r"[a-z0-9]+")
HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
MIN_KEYWORD_LENGTH = 2
EMBEDDING_BATCH_SIZE = 100

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_documents(cleaned_dir: Path = CLEANED_DIR) -> List[Document]:
    """
    Load cleaned text files into LangChain documents.

    Parameters
    ----------
    cleaned_dir : Path
        Directory containing cleaned ``.txt`` files.

    Returns
    -------
    list of Document
        One document per non-empty file, tagged with its source filename.
    """
    documents: List[Document] = []
    for path in sorted(cleaned_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(page_content=text, metadata={"source": path.name}))

    return documents


def extract_keywords(text: str) -> List[str]:
    """
    Tokenize text into lowercase content keywords.

    Tokens are kept when they meet the minimum length and are not English
    stop words. Repeated tokens are preserved in order so that BM25 term
    frequency reflects the natural occurrence of each keyword.

    Parameters
    ----------
    text : str
        Raw chunk or query text.

    Returns
    -------
    list of str
        Content-bearing keyword tokens, including repeats.
    """
    keywords: List[str] = []
    for token in WORD_PATTERN.findall(text.lower()):
        if len(token) < MIN_KEYWORD_LENGTH:
            continue
        if token in ENGLISH_STOP_WORDS:
            continue
        keywords.append(token)

    return keywords


def document_url(source: str) -> str:
    """
    Reconstruct a page URL from a cleaned text filename.

    Parameters
    ----------
    source : str
        Cleaned text filename, such as ``education__masters-programs.txt``.

    Returns
    -------
    str
        Absolute page URL with a trailing slash.
    """
    stem = source[:-4] if source.endswith(".txt") else source
    if not stem or stem == "index":
        return f"{BASE_URL}/"

    path = stem.replace("__", "/")
    return f"{BASE_URL}/{path}/"


def split_sections(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split markdown-style page text into a title and titled sections.

    The first level-one heading is treated as the page title. Every heading
    starts a new section whose body is the text up to the next heading.

    Parameters
    ----------
    text : str
        Section-aware page text with markdown-style headings.

    Returns
    -------
    tuple
        The page title and a list of ``(section_title, section_body)`` pairs
        with empty bodies removed.
    """
    page_title = ""
    sections: List[Tuple[str, str]] = []

    parts = HEADING_LINE.split(text)
    for hashes, heading, body in zip(parts[1::3], parts[2::3], parts[3::3]):
        heading = heading.strip()
        body = body.strip()

        if len(hashes) == 1 and not page_title:
            page_title = heading

        if body:
            sections.append((heading, body))

    return page_title, sections


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """
    Split documents into a table of chunks with rich metadata.

    Parameters
    ----------
    documents : list of Document
        Source documents to split.
    chunk_size : int
        Target maximum tokens per chunk.
    chunk_overlap : int
        Tokens shared between consecutive chunks.

    Returns
    -------
    pandas.DataFrame
        Table with a ``metadata`` column (page title, section title,
        keywords, and URL) and a ``chunk`` column of chunk text.
    """
    splitter = TokenTextSplitter.from_tiktoken_encoder(
        encoding_name=TIKTOKEN_ENCODING,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    rows = []
    for document in documents:
        source = document.metadata.get("source", "")
        url = document_url(source)
        page_title, sections = split_sections(document.page_content)

        for section_title, body in sections:
            for chunk in splitter.split_text(body):
                metadata = {
                    "page_title": page_title,
                    "section_title": section_title,
                    "keywords": extract_keywords(chunk),
                    "url": url,
                }
                rows.append({"metadata": metadata, "chunk": chunk})

    return pd.DataFrame(rows, columns=["metadata", "chunk"])


def embed_texts(texts: List[str], client: OpenAI = client) -> List[List[float]]:
    """
    Embed texts with the OpenAI client.

    Parameters
    ----------
    texts : list of str
        Texts to embed.
    client : OpenAI
        OpenAI client used to create embeddings.

    Returns
    -------
    list of list of float
        One embedding vector per input text, in order.
    """
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

    return vectors


def build_vector_store(chunks: pd.DataFrame, client: OpenAI = client) -> FAISS:
    """
    Embed chunks with the OpenAI client and index them in FAISS.

    Parameters
    ----------
    chunks : pandas.DataFrame
        Chunk table with ``chunk`` and ``metadata`` columns.
    client : OpenAI
        OpenAI client used to create text embeddings.

    Returns
    -------
    FAISS
        In-memory FAISS vector store over the chunk embeddings.
    """
    texts = chunks["chunk"].tolist()
    vectors = embed_texts(texts, client)

    return FAISS.from_embeddings(
        text_embeddings=zip(texts, vectors),
        embedding=lambda text: embed_texts([text], client)[0],
        metadatas=chunks["metadata"].tolist(),
    )


def save_vector_store(vector_store: FAISS, directory: Path = VECTOR_STORE_DIR) -> Path:
    """
    Persist a FAISS vector store to disk.

    Parameters
    ----------
    vector_store : FAISS
        Vector store to save.
    directory : Path
        Destination directory for the FAISS index files.

    Returns
    -------
    Path
        Directory the vector store was written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(directory))
    return directory


def load_vector_store(client: OpenAI = client, directory: Path = VECTOR_STORE_DIR) -> FAISS:
    """
    Load a persisted FAISS vector store from disk.

    Parameters
    ----------
    client : OpenAI
        OpenAI client used to embed queries against the store.
    directory : Path
        Directory containing the saved FAISS index files.

    Returns
    -------
    FAISS
        Reconstructed FAISS vector store.
    """
    return FAISS.load_local(
        str(directory),
        lambda text: embed_texts([text], client)[0],
        allow_dangerous_deserialization=True,
    )


def build_embeddings(client: OpenAI = client) -> FAISS:
    """
    Build and persist the FAISS vector store from cleaned documents.

    Parameters
    ----------
    client : OpenAI
        OpenAI client used to create text embeddings.

    Returns
    -------
    FAISS
        The saved FAISS vector store.
    """
    documents = load_documents()
    chunk_table = chunk_documents(documents)
    vector_store = build_vector_store(chunk_table, client)
    save_vector_store(vector_store)
    return vector_store


if __name__ == "__main__":
    build_embeddings()

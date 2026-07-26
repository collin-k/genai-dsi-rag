"""
Hybrid retrieval for the UChicago MS-ADS RAG assistant.

Combines:
1. FAISS semantic retrieval
2. BM25 keyword retrieval
3. Weighted Reciprocal Rank Fusion (RRF)

Expected project files:
    data/vector_store/index.faiss
    data/vector_store/index.pkl

Expected environment variable in the project-root .env file:
    OPENAI_API_KEY=your_actual_openai_key
"""

import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    WORD_PATTERN,
    MIN_KEYWORD_LENGTH,
    EMBEDDING_BATCH_SIZE,
    DEFAULT_RRF_CONSTANT,
    DEFAULT_CANDIDATE_K,
)

try:
    # Used when imported as part of the src package.
    from .config import (
        EMBEDDING_MODEL,
        HYBRID_WEIGHTS,
        RETRIEVER_TOP_K,
        VECTOR_STORE_DIR,
    )
except ImportError:
    # Used when run directly: python src/hybrid_retrieval.py
    from config import (
        EMBEDDING_MODEL,
        HYBRID_WEIGHTS,
        RETRIEVER_TOP_K,
        VECTOR_STORE_DIR,
    )


def find_project_root() -> Path:
    """Return the project root containing the src directory."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n\n"
        f"Create this file:\n{PROJECT_ROOT / '.env'}\n\n"
        "Add this line:\nOPENAI_API_KEY=your_actual_openai_key"
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class OpenAIEmbeddingAdapter(Embeddings):
    """LangChain-compatible wrapper for OpenAI embeddings."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in batches."""
        vectors: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed one user query."""
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return response.data[0].embedding


embedding_model = OpenAIEmbeddingAdapter(
    client=openai_client,
    model=EMBEDDING_MODEL,
)


def tokenize_text(text: str) -> List[str]:
    """Convert text into lowercase BM25 keyword tokens."""
    if not isinstance(text, str):
        return []

    return [
        token
        for token in WORD_PATTERN.findall(text.lower())
        if len(token) >= MIN_KEYWORD_LENGTH
        and token not in ENGLISH_STOP_WORDS
    ]


def create_document_id(document: Document) -> str:
    """Create a stable ID for a document chunk."""
    url = document.metadata.get("url", "")
    section = document.metadata.get("section_title", "")
    raw = f"{url}::{section}::{document.page_content.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_saved_faiss(directory: Path = VECTOR_STORE_DIR) -> FAISS:
    """Load the FAISS index created by embeddings.py."""
    directory = Path(directory)
    faiss_file = directory / "index.faiss"
    metadata_file = directory / "index.pkl"

    missing = [
        str(path)
        for path in (faiss_file, metadata_file)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The saved FAISS vector store is incomplete.\n\n"
            "Missing:\n- "
            + "\n- ".join(missing)
            + "\n\nAsk your teammate for the data/vector_store folder "
              "or run src/embeddings.py once to create it."
        )

    return FAISS.load_local(
        folder_path=str(directory),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def get_stored_documents(vector_store: FAISS) -> List[Document]:
    """Extract all LangChain documents stored inside FAISS."""
    documents = [
        document
        for document in vector_store.docstore._dict.values()
        if isinstance(document, Document)
        and document.page_content.strip()
    ]

    if not documents:
        raise ValueError(
            "No document chunks were found inside the saved FAISS store."
        )

    return documents


class HybridRetriever:
    """Combine FAISS and BM25 rankings with weighted RRF."""

    def __init__(
        self,
        vector_store: FAISS,
        dense_k: int = DEFAULT_CANDIDATE_K,
        sparse_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = RETRIEVER_TOP_K,
        dense_weight: float = HYBRID_WEIGHTS[0],
        sparse_weight: float = HYBRID_WEIGHTS[1],
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
    ) -> None:
        if min(dense_k, sparse_k, final_k) < 1:
            raise ValueError("dense_k, sparse_k, and final_k must be at least 1.")

        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("Hybrid weights cannot be negative.")

        if dense_weight + sparse_weight == 0:
            raise ValueError("At least one hybrid weight must be positive.")

        self.vector_store = vector_store
        self.documents = get_stored_documents(vector_store)
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.final_k = final_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_constant = rrf_constant

        tokenized_corpus = [
            tokenize_text(document.page_content)
            for document in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def dense_search(self, query: str) -> List[Dict[str, Any]]:
        """Run semantic search with FAISS."""
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.dense_k,
        )

        return [
            {
                "document": document,
                "rank": rank,
                "raw_score": float(score),
            }
            for rank, (document, score) in enumerate(results, start=1)
        ]

    def sparse_search(self, query: str) -> List[Dict[str, Any]]:
        """Run lexical keyword search with BM25."""
        query_tokens = tokenize_text(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: List[Dict[str, Any]] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score <= 0:
                continue

            results.append(
                {
                    "document": self.documents[index],
                    "rank": len(results) + 1,
                    "raw_score": score,
                }
            )

            if len(results) >= self.sparse_k:
                break

        return results

    def retrieve_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """Fuse FAISS and BM25 rankings using weighted RRF."""
        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("The query cannot be empty.")

        dense_results = self.dense_search(clean_query)
        sparse_results = self.sparse_search(clean_query)

        fused_scores = defaultdict(float)
        documents: Dict[str, Document] = {}
        details = defaultdict(dict)

        for result in dense_results:
            document = result["document"]
            doc_id = create_document_id(document)
            rank = result["rank"]
            rrf_score = self.dense_weight / (self.rrf_constant + rank)

            documents[doc_id] = document
            fused_scores[doc_id] += rrf_score
            details[doc_id]["dense_rank"] = rank
            details[doc_id]["faiss_score"] = result["raw_score"]
            details[doc_id]["dense_rrf"] = rrf_score

        for result in sparse_results:
            document = result["document"]
            doc_id = create_document_id(document)
            rank = result["rank"]
            rrf_score = self.sparse_weight / (self.rrf_constant + rank)

            documents[doc_id] = document
            fused_scores[doc_id] += rrf_score
            details[doc_id]["bm25_rank"] = rank
            details[doc_id]["bm25_score"] = result["raw_score"]
            details[doc_id]["sparse_rrf"] = rrf_score

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        return [
            {
                "rank": rank,
                "document": documents[doc_id],
                "hybrid_score": float(fused_scores[doc_id]),
                "details": dict(details[doc_id]),
            }
            for rank, doc_id in enumerate(
                ranked_ids[: self.final_k],
                start=1,
            )
        ]

    def invoke(self, query: str) -> List[Document]:
        """Return only the final LangChain Document objects."""
        return [
            result["document"]
            for result in self.retrieve_with_scores(query)
        ]


def create_hybrid_retriever() -> HybridRetriever:
    """Load FAISS and create the hybrid retriever."""
    return HybridRetriever(
        vector_store=load_saved_faiss(),
        dense_k=DEFAULT_CANDIDATE_K,
        sparse_k=DEFAULT_CANDIDATE_K,
        final_k=RETRIEVER_TOP_K,
        dense_weight=HYBRID_WEIGHTS[0],
        sparse_weight=HYBRID_WEIGHTS[1],
        rrf_constant=DEFAULT_RRF_CONSTANT,
    )


def display_results(results: List[Dict[str, Any]]) -> None:
    """Print retrieval results for a command-line test."""
    if not results:
        print("No results were found.")
        return

    for result in results:
        document = result["document"]

        print("\n" + "=" * 80)
        print("Hybrid rank:", result["rank"])
        print("Hybrid score:", round(result["hybrid_score"], 6))
        print("Retrieval details:", result["details"])
        print("Page title:", document.metadata.get("page_title", "Unknown"))
        print("Section:", document.metadata.get("section_title", "Unknown"))
        print("URL:", document.metadata.get("url", "Unknown"))
        print("\nContent:")
        print(document.page_content[:800])


if __name__ == "__main__":
    print("Loading the saved FAISS vector store...")
    retriever = create_hybrid_retriever()
    print("Hybrid retriever created successfully.")

    test_question = (
        "What options are available for completing "
        "the MS in Applied Data Science?"
    )

    print("\nQuestion:")
    print(test_question)

    display_results(
        retriever.retrieve_with_scores(test_question)
    )

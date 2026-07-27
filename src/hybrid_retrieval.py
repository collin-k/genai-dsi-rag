"""
Hybrid retrieval for the UChicago MS-ADS RAG assistant.

Pipeline:
1. FAISS semantic retrieval
2. BM25 keyword retrieval
3. Weighted Reciprocal Rank Fusion (RRF)
4. OpenAI listwise reranking

Expected project files:
    data/vector_store/index.faiss
    data/vector_store/index.pkl

Expected environment variable in the project-root .env file:
    OPENAI_API_KEY=your_actual_openai_key
"""

from __future__ import annotations

import hashlib
import json
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

try:
    from .config import (
        DEFAULT_CANDIDATE_K,
        DEFAULT_RRF_CONSTANT,
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_MODEL,
        HYBRID_WEIGHTS,
        MIN_KEYWORD_LENGTH,
        RERANK_CANDIDATE_K,
        RERANK_ENABLED,
        RERANKER_MODEL,
        RETRIEVER_TOP_K,
        VECTOR_STORE_DIR,
        WORD_PATTERN,
    )
except ImportError:
    from config import (
        DEFAULT_CANDIDATE_K,
        DEFAULT_RRF_CONSTANT,
        EMBEDDING_BATCH_SIZE,
        EMBEDDING_MODEL,
        HYBRID_WEIGHTS,
        MIN_KEYWORD_LENGTH,
        RERANK_CANDIDATE_K,
        RERANK_ENABLED,
        RERANKER_MODEL,
        RETRIEVER_TOP_K,
        VECTOR_STORE_DIR,
        WORD_PATTERN,
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


def document_search_text(document: Document) -> str:
    """
    Build searchable text that includes page and section titles.

    Parameters
    ----------
    document : Document
        Chunk document from the FAISS store.

    Returns
    -------
    str
        Title-enriched text used for BM25 and reranking.
    """
    metadata = document.metadata or {}
    page_title = str(metadata.get("page_title", "")).strip()
    section_title = str(metadata.get("section_title", "")).strip()
    parts = [part for part in (page_title, section_title, document.page_content) if part]
    return "\n".join(parts)


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


def extract_json_payload(text: str) -> Any:
    """
    Parse a JSON object or array from a model response.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    Any
        Parsed JSON value.
    """
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Reranker did not return valid JSON: {text}")
        return json.loads(match.group(1))


def listwise_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    client: OpenAI = openai_client,
    model: str = RERANKER_MODEL,
) -> List[Dict[str, Any]]:
    """
    Reorder fused candidate chunks with an OpenAI listwise ranking prompt.

    Parameters
    ----------
    query : str
        User question.
    candidates : list of dict
        Hybrid retrieval candidates with document metadata.
    top_k : int
        Number of chunks to keep after reranking.
    client : OpenAI
        OpenAI client used for the ranking call.
    model : str
        Model name used for listwise ranking.

    Returns
    -------
    list of dict
        Reranked candidates truncated to ``top_k``.
    """
    if not candidates:
        return []

    if len(candidates) == 1 or top_k <= 0:
        return candidates[: max(top_k, 0)]

    prompt_blocks = []
    for index, candidate in enumerate(candidates, start=1):
        document = candidate["document"]
        metadata = document.metadata or {}
        preview = document_search_text(document)[:1200]
        prompt_blocks.append(
            f"[Candidate {index}]\n"
            f"Page: {metadata.get('page_title', 'Unknown')}\n"
            f"Section: {metadata.get('section_title', 'Unknown')}\n"
            f"URL: {metadata.get('url', 'Unknown')}\n"
            f"Content:\n{preview}"
        )

        prompt = f"""
Rank these retrieved MS-ADS website passages by how well they answer the question.

Prefer passages that:
- directly answer the question
- contain specific facts such as course names, deadlines, scores, addresses, or policies
- come from the most relevant page/section

Question:
{query}

Candidates:
{chr(10).join(prompt_blocks)}

Return JSON only in this format:
{{"ranking": [1, 3, 2]}}

Use each candidate index at most once. Rank from most relevant to least relevant.
""".strip()

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    payload = extract_json_payload(response.output_text)
    ranking = payload.get("ranking", payload) if isinstance(payload, dict) else payload

    if not isinstance(ranking, list):
        return candidates[:top_k]

    ordered: List[Dict[str, Any]] = []
    seen = set()

    for raw_index in ranking:
        try:
            index = int(raw_index) - 1
        except (TypeError, ValueError):
            continue

        if index < 0 or index >= len(candidates) or index in seen:
            continue

        seen.add(index)
        candidate = dict(candidates[index])
        details = dict(candidate.get("details", {}))
        details["rerank_position"] = len(ordered) + 1
        details["pre_rerank_rank"] = candidate.get("rank")
        candidate["details"] = details
        ordered.append(candidate)

    for index, candidate in enumerate(candidates):
        if index in seen:
            continue
        fallback = dict(candidate)
        details = dict(fallback.get("details", {}))
        details["rerank_position"] = len(ordered) + 1
        details["pre_rerank_rank"] = fallback.get("rank")
        fallback["details"] = details
        ordered.append(fallback)

    return ordered[:top_k]


class HybridRetriever:
    """Combine FAISS and BM25 rankings with weighted RRF and optional reranking."""

    def __init__(
        self,
        vector_store: FAISS,
        dense_k: int = DEFAULT_CANDIDATE_K,
        sparse_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = RETRIEVER_TOP_K,
        dense_weight: float = HYBRID_WEIGHTS[0],
        sparse_weight: float = HYBRID_WEIGHTS[1],
        rrf_constant: int = DEFAULT_RRF_CONSTANT,
        rerank_candidate_k: int = RERANK_CANDIDATE_K,
        rerank_enabled: bool = RERANK_ENABLED,
    ) -> None:
        if min(dense_k, sparse_k, final_k, rerank_candidate_k) < 1:
            raise ValueError(
                "dense_k, sparse_k, final_k, and rerank_candidate_k must be at least 1."
            )

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
        self.rerank_candidate_k = max(rerank_candidate_k, final_k)
        self.rerank_enabled = rerank_enabled

        tokenized_corpus = [
            tokenize_text(document_search_text(document))
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

    def fuse_rankings(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Fuse FAISS and BM25 rankings using weighted RRF.

        Parameters
        ----------
        dense_results : list of dict
            Dense retrieval results.
        sparse_results : list of dict
            Sparse retrieval results.
        limit : int
            Maximum number of fused candidates to return.

        Returns
        -------
        list of dict
            Fused ranking with hybrid scores and per-retriever details.
        """
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
                ranked_ids[:limit],
                start=1,
            )
        ]

    def retrieve_with_scores(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve, fuse, and optionally rerank supporting passages."""
        if not isinstance(query, str):
            raise TypeError("The query must be a string.")

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("The query cannot be empty.")

        dense_results = self.dense_search(clean_query)
        sparse_results = self.sparse_search(clean_query)
        fused_results = self.fuse_rankings(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=self.rerank_candidate_k if self.rerank_enabled else self.final_k,
        )

        if self.rerank_enabled:
            ranked_results = listwise_rerank(
                query=clean_query,
                candidates=fused_results,
                top_k=self.final_k,
            )
        else:
            ranked_results = fused_results[: self.final_k]

        return [
            {
                "rank": rank,
                "document": result["document"],
                "hybrid_score": float(result["hybrid_score"]),
                "details": dict(result.get("details", {})),
            }
            for rank, result in enumerate(ranked_results, start=1)
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
        rerank_candidate_k=RERANK_CANDIDATE_K,
        rerank_enabled=RERANK_ENABLED,
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

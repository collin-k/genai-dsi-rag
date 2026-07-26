"""
Grounded LLM layer for the UChicago MS-ADS RAG assistant.

Pipeline:
    User question
    -> Hybrid retriever
    -> Retrieved context
    -> OpenAI LLM
    -> Grounded answer with sources

Run from the project root:
    python -m src.rag
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from config import SYSTEM_INSTRUCTIONS, LLM_MODEL, MAX_CONTEXT_CHARACTERS

try:
    from .hybrid_retrieval import create_hybrid_retriever
except ImportError:
    from hybrid_retrieval import create_hybrid_retriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found.\n\n"
        f"Add it to:\n{PROJECT_ROOT / '.env'}"
    )

client = OpenAI(api_key=OPENAI_API_KEY)


def format_context(
    retrieval_results: List[Dict[str, Any]],
) -> str:
    """Format retrieved chunks for the language model."""
    sections: List[str] = []

    for result in retrieval_results:
        document = result["document"]
        metadata = document.metadata or {}

        rank = result["rank"]
        page_title = metadata.get("page_title", "Unknown page")
        section_title = metadata.get("section_title", "Unknown section")
        url = metadata.get("url", "Unknown URL")

        sections.append(
            f"[Retrieved Source {rank}]\n"
            f"Page title: {page_title}\n"
            f"Section: {section_title}\n"
            f"URL: {url}\n"
            f"Content:\n{document.page_content.strip()}"
        )

    return "\n\n".join(sections)[:MAX_CONTEXT_CHARACTERS]


def unique_sources(
    retrieval_results: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return unique source URLs from retrieved chunks."""
    sources: List[Dict[str, str]] = []
    seen_urls = set()

    for result in retrieval_results:
        document = result["document"]
        metadata = document.metadata or {}

        url = metadata.get("url", "")
        page_title = metadata.get("page_title", "DSI webpage")
        section_title = metadata.get("section_title", "")

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        sources.append(
            {
                "page_title": page_title,
                "section_title": section_title,
                "url": url,
            }
        )

    return sources


class DsiRagAssistant:
    """Retrieve supporting passages and generate grounded answers."""

    def __init__(self) -> None:
        self.retriever = create_hybrid_retriever()

    def answer(
        self,
        question: str,
    ) -> Dict[str, Any]:
        """Answer one question using hybrid retrieval and OpenAI."""
        if not isinstance(question, str):
            raise TypeError("Question must be a string.")

        clean_question = question.strip()

        if not clean_question:
            raise ValueError("Question cannot be empty.")

        retrieval_results = self.retriever.retrieve_with_scores(
            clean_question
        )

        context = format_context(retrieval_results)

        user_prompt = f"""
Retrieved MS-ADS website context:

{context}

User question:

{clean_question}

Produce a grounded answer using only the retrieved context.
""".strip()

        response = client.responses.create(
            model=LLM_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=user_prompt,
        )

        answer_text = response.output_text.strip()

        return {
            "question": clean_question,
            "answer": answer_text,
            "sources": unique_sources(retrieval_results),
            "retrieval_results": retrieval_results,
        }


def display_answer(
    result: Dict[str, Any],
) -> None:
    """Print the answer and sources."""
    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result["question"])

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\n" + "=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for index, source in enumerate(result["sources"], start=1):
        print(f"{index}. {source['page_title']}")

        if source["section_title"]:
            print(f"   Section: {source['section_title']}")

        print(f"   {source['url']}")


if __name__ == "__main__":
    assistant = DsiRagAssistant()

    test_question = (
        "How many courses must students complete "
        "to earn the MS in Applied Data Science?"
    )

    result = assistant.answer(test_question)
    display_answer(result)

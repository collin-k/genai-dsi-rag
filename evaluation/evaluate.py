"""Evaluate the UChicago MS-ADS hybrid RAG system.

Run from the project root:
    python evaluation/evaluate.py

Expected files:
    evaluation/evaluation_set.json
    data/vector_store/index.faiss
    data/vector_store/index.pkl
    .env containing OPENAI_API_KEY
"""

import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVALUATION_FILE = EVALUATION_DIR / "evaluation_set.json"
RESULTS_JSON = EVALUATION_DIR / "evaluation_results.json"
RESULTS_CSV = EVALUATION_DIR / "evaluation_results.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid_retrieval import create_hybrid_retriever

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
USE_LLM_JUDGE = True
MAX_CONTEXT_CHARACTERS = 14000

load_dotenv(PROJECT_ROOT / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found in the project-root .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def load_evaluation_set(path: Path = EVALUATION_FILE) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {path}\n"
            "Copy evaluation_set.example.json to evaluation_set.json and replace "
            "the sample questions with the instructor's evaluation set."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("evaluation_set.json must contain a non-empty JSON list.")

    for index, item in enumerate(data, start=1):
        for field in ("id", "question", "ground_truth"):
            if field not in item:
                raise ValueError(f"Item {index} is missing required field: {field}")
        item.setdefault("expected_keywords", [])
        item.setdefault("expected_url_fragments", [])

    return data


def format_context(documents: Sequence[Any]) -> str:
    sections: List[str] = []

    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        sections.append(
            f"[Source {index}]\n"
            f"Page: {metadata.get('page_title', 'Unknown page')}\n"
            f"Section: {metadata.get('section_title', 'Unknown section')}\n"
            f"URL: {metadata.get('url', 'Unknown URL')}\n"
            f"Content:\n{document.page_content.strip()}"
        )

    return "\n\n".join(sections)[:MAX_CONTEXT_CHARACTERS]


def generate_answer(question: str, documents: Sequence[Any]) -> str:
    context = format_context(documents)
    prompt = f"""
You are a University of Chicago MS in Applied Data Science program assistant.
Answer the question using only the retrieved context.

Rules:
- Do not use outside knowledge.
- If the answer is unavailable, say so clearly.
- Be concise but complete.
- Include relevant source URLs when possible.

Retrieved context:
{context}

Question:
{question}
""".strip()

    response = client.responses.create(model=ANSWER_MODEL, input=prompt)
    return response.output_text.strip()


def expected_url_rank(documents: Sequence[Any], fragments: Sequence[str]) -> int:
    cleaned = [normalize_text(value) for value in fragments if str(value).strip()]
    if not cleaned:
        return 0

    for rank, document in enumerate(documents, start=1):
        url = normalize_text((document.metadata or {}).get("url", ""))
        if any(fragment in url for fragment in cleaned):
            return rank
    return 0


def keyword_recall(text: str, expected_keywords: Sequence[str]) -> float:
    keywords = [normalize_text(value) for value in expected_keywords if str(value).strip()]
    if not keywords:
        return 0.0

    normalized = normalize_text(text)
    return sum(keyword in normalized for keyword in keywords) / len(keywords)


def token_f1(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction)
    expected = tokenize(reference)
    if not predicted or not expected:
        return 0.0

    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Judge did not return valid JSON: {text}")
        return json.loads(match.group(0))


def llm_judge(question: str, ground_truth: str, answer: str, context: str) -> Dict[str, Any]:
    prompt = f"""
Evaluate this retrieval-augmented answer using integer scores from 1 to 5.

relevance:
1 = unrelated or incorrect
5 = directly answers the question and agrees with the ground-truth facts

faithfulness:
1 = mostly unsupported by retrieved context
5 = every important claim is supported by retrieved context

Return JSON only:
{{"relevance": 1, "faithfulness": 1, "reason": "brief explanation"}}

Question:
{question}

Ground truth:
{ground_truth}

Generated answer:
{answer}

Retrieved context:
{context}
""".strip()

    response = client.responses.create(model=JUDGE_MODEL, input=prompt)
    result = extract_json_object(response.output_text)
    result["relevance"] = max(1, min(5, int(result.get("relevance", 1))))
    result["faithfulness"] = max(1, min(5, int(result.get("faithfulness", 1))))
    result["reason"] = str(result.get("reason", "")).strip()
    return result


def evaluate_item(item: Dict[str, Any], retriever: Any) -> Dict[str, Any]:
    retrieval_results = retriever.retrieve_with_scores(item["question"])
    documents = [result["document"] for result in retrieval_results]
    context = format_context(documents)
    answer = generate_answer(item["question"], documents)

    first_rank = expected_url_rank(documents, item["expected_url_fragments"])
    hit_at_k = 1 if first_rank else 0
    reciprocal_rank = 1 / first_rank if first_rank else 0.0

    judge = {"relevance": None, "faithfulness": None, "reason": ""}
    if USE_LLM_JUDGE:
        judge = llm_judge(item["question"], item["ground_truth"], answer, context)

    return {
        "id": item["id"],
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "generated_answer": answer,
        "hit_at_k": hit_at_k,
        "first_relevant_rank": first_rank,
        "reciprocal_rank": round(reciprocal_rank, 4),
        "context_keyword_recall": round(keyword_recall(context, item["expected_keywords"]), 4),
        "answer_keyword_recall": round(keyword_recall(answer, item["expected_keywords"]), 4),
        "answer_token_f1": round(token_f1(answer, item["ground_truth"]), 4),
        "response_relevance_1_to_5": judge["relevance"],
        "faithfulness_1_to_5": judge["faithfulness"],
        "judge_reason": judge["reason"],
        "retrieved_sources": [
            {
                "rank": result["rank"],
                "url": result["document"].metadata.get("url", ""),
                "page_title": result["document"].metadata.get("page_title", ""),
                "section_title": result["document"].metadata.get("section_title", ""),
                "hybrid_score": round(result["hybrid_score"], 8),
                "details": result["details"],
            }
            for result in retrieval_results
        ],
    }


def average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    relevance = [
        float(item["response_relevance_1_to_5"])
        for item in results
        if item["response_relevance_1_to_5"] is not None
    ]
    faithfulness = [
        float(item["faithfulness_1_to_5"])
        for item in results
        if item["faithfulness_1_to_5"] is not None
    ]

    return {
        "number_of_questions": len(results),
        "retrieval_hit_rate_at_k": round(average([item["hit_at_k"] for item in results]), 4),
        "mean_reciprocal_rank": round(average([item["reciprocal_rank"] for item in results]), 4),
        "mean_context_keyword_recall": round(average([item["context_keyword_recall"] for item in results]), 4),
        "mean_answer_keyword_recall": round(average([item["answer_keyword_recall"] for item in results]), 4),
        "mean_answer_token_f1": round(average([item["answer_token_f1"] for item in results]), 4),
        "mean_response_relevance_1_to_5": round(average(relevance), 4) if relevance else None,
        "mean_faithfulness_1_to_5": round(average(faithfulness), 4) if faithfulness else None,
    }


def save_results(results: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "results": list(results)}, indent=2),
        encoding="utf-8",
    )

    fields = [
        "id", "question", "ground_truth", "generated_answer", "hit_at_k",
        "first_relevant_rank", "reciprocal_rank", "context_keyword_recall",
        "answer_keyword_recall", "answer_token_f1", "response_relevance_1_to_5",
        "faithfulness_1_to_5", "judge_reason", "retrieved_urls",
    ]

    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field, "") for field in fields}
            row["retrieved_urls"] = " | ".join(
                source["url"] for source in result["retrieved_sources"]
            )
            writer.writerow(row)


def main() -> None:
    evaluation_set = load_evaluation_set()
    retriever = create_hybrid_retriever()
    results: List[Dict[str, Any]] = []

    print(f"Evaluating {len(evaluation_set)} questions...\n")

    for index, item in enumerate(evaluation_set, start=1):
        print(f"[{index}/{len(evaluation_set)}] {item['question']}")
        try:
            result = evaluate_item(item, retriever)
            results.append(result)
            print(
                f"  Hit@K={result['hit_at_k']} | "
                f"RR={result['reciprocal_rank']} | "
                f"Answer keyword recall={result['answer_keyword_recall']} | "
                f"Relevance={result['response_relevance_1_to_5']}"
            )
        except Exception as error:
            print(f"  ERROR: {error}")
            results.append({
                "id": item["id"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "generated_answer": "",
                "hit_at_k": 0,
                "first_relevant_rank": 0,
                "reciprocal_rank": 0.0,
                "context_keyword_recall": 0.0,
                "answer_keyword_recall": 0.0,
                "answer_token_f1": 0.0,
                "response_relevance_1_to_5": None,
                "faithfulness_1_to_5": None,
                "judge_reason": f"Evaluation error: {error}",
                "retrieved_sources": [],
            })

    summary = summarize(results)
    save_results(results, summary)

    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print(f"\nDetailed JSON: {RESULTS_JSON}")
    print(f"CSV report:    {RESULTS_CSV}")


if __name__ == "__main__":
    main()

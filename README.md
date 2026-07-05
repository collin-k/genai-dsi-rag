# MS in Applied Data Science — RAG Assistant

A Retrieval-Augmented Generation (RAG) conversational assistant that helps prospective students, current enrollees, and alumni get accurate, context-aware answers about the [MS in Applied Data Science program](https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/) at the University of Chicago.

## Overview

The MS in Applied Data Science program offers a comprehensive curriculum covering both theoretical and practical aspects of data science. Program information spans course offerings, faculty expertise, admissions requirements, career outcomes, and more — spread across many pages on the program website. Navigating that content can be time-consuming and easy to miss.

This repository builds a RAG-based system that combines semantic search over program content with a large language model, so users can ask questions in natural language and receive grounded, relevant responses.

## Goals

- Extract and structure textual content from the MS in Applied Data Science program website and its subpages
- Enable fast, accurate retrieval of program-related information through vector search
- Generate conversational answers grounded in retrieved context
- Provide an intuitive chat interface for interactive Q&A
- Evaluate retrieval accuracy and response relevance against a defined test set

## System Components

At a high level, the system is organized around four areas:

| Area | Purpose |
|------|---------|
| **Data preparation** | Collect and clean content from the program website into a structured knowledge base |
| **Retrieval** | Generate embeddings and store them in a vector database for semantic search |
| **Generation** | Use an LLM with retrieved context to produce accurate, cited answers |
| **Interface** | Chat UI for users to ask questions and view responses |

Responsible AI practices — grounding answers in retrieved content, handling out-of-scope queries, and limiting hallucination — are part of the design.

## Example Interactions

**Core courses**

> **Q:** What are the core courses in the MS in Applied Data Science program?
>
> **A:** The core courses in the MS in Applied Data Science program include Machine Learning, Data Engineering Platforms, Statistical Inference, and Applied Data Science.

**Admissions**

> **Q:** What are the admission requirements for the MS in Applied Data Science program?
>
> **A:** Applicants need a bachelor's degree in a related field, with coursework in programming, statistics, and mathematics. The application also requires a personal statement, letters of recommendation, and a resume.

**Capstone**

> **Q:** Can you provide information about the capstone project?
>
> **A:** The capstone project is a key component of the MS in Applied Data Science program, where students work on real-world problems, applying their learned skills to develop data-driven solutions.

## Repository Structure

```
genai-dsi-rag/
├── src/                  # Application source code
├── data/
│   └── kb/               # Cleaned knowledge-base text files
├── evaluation/           # Test queries and evaluation scripts
├── docs/                 # System documentation
├── presentation/         # Slide deck
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

**Requirements:** Python 3.9+

1. Clone the repository and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Add your API keys to `.env` (see `.env.example` for required variables).

## Deliverables

- A functional RAG chatbot for MS in Applied Data Science program inquiries
- Documentation covering preprocessing, architecture, and system design (5+ pages)
- A user-friendly conversational interface
- A presentation on implementation, challenges, and future improvements (~10 minutes)
- Evaluation metrics for retrieval accuracy and response relevance
"""Test queries for evaluating DSI RAG retrieval and response quality."""

TEST_CASES = [
    {
        "query": "What are the core courses in the MS in Applied Data Science program?",
        "expected_answer": "The core courses in the MS in Applied Data Science program include Machine Learning, Data Engineering Platforms, Statistical Inference, and Applied Data Science.",
    },
    {
        "query": "What are the admission requirements for the MS in Applied Data Science program?",
        "expected_answer": "Applicants need a bachelor's degree in a related field, with coursework in programming, statistics, and mathematics. The application also requires a personal statement, letters of recommendation, and a resume.",
    },
    {
        "query": "Can you provide information about the capstone project?",
        "expected_answer": "TThe capstone project is a key component of the MS in Applied Data Science program, where students work on real-world problems, applying their learned skills to develop data-driven solutions.",
    }
]

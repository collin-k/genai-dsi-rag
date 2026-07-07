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
        "expected_answer": "The capstone project is a key component of the MS in Applied Data Science program, where students work on real-world problems, applying their learned skills to develop data-driven solutions.",
    },
    {
        "query": "What scholarships are available for the program?",
        "expected_answer": "The Data Science Institute Scholarship, MS in Applied Data Science Alumni Scholarship, etc.",
    },
    {
        "query": "What are the minimum scores for the TOEFL and IELTS English Language Requirement?",
        "expected_answer": "Minimum scores for the Master's in Applied Data Science program: TOEFL, 102 (no subscore requirement); IELTS, 7 (no subscore requirement).",
    },
    {
        "query": "Is there an application fee waiver?",
        "expected_answer": "For questions regarding an application fee waiver, please refer to the Physical Sciences Division fee waiver policy.",
    },
    {
        "query": "What are the deadlines for the in-person program?",
        "expected_answer": "In-person program deadlines include: Priority Application Deadline, Scholarship Priority Deadline, International Application Deadline (requiring visa sponsorship from UChicago), Second Priority Application Deadline, Third Priority Application Deadline, and Final Application Deadline.",
    },
    {
        "query": "How long will it take for me to receive a decision on my application?",
        "expected_answer": "In-Person application decisions are released approximately 1 to 2 months after each respected deadline. Online application decisions are released on a rolling basis.",
    },
    {
        "query": "Can I set up an advising appointment with the enrollment management team?",
        "expected_answer": "Yes, meet your admissions counselor by scheduling an appointment at https://apply-psd.uchicago.edu/portal/applied-data-science.",
    },
    {
        "query": "Where can I mail my official transcripts?",
        "expected_answer": "The University of Chicago, Attention: MS in Applied Data Science Admissions, 455 N Cityfront Plaza Dr., Suite 950, Chicago, Illinois 60611.",
    },
    {
        "query": "Does the Master's in Applied Data Science Online program provide visa sponsorship?",
        "expected_answer": "Only our In-Person, Full-Time program is Visa eligible.",
    },
    {
        "query": "How do I apply to the MBA/MS program?",
        "expected_answer": "Applicants interested in the Joint MBA/MS degree will apply through Booth's centralized, joint-application process. Applicants should complete the Chicago Booth Full-Time MBA application and select the MBA/MS in Applied Data Science as their program of interest.",
    },
    {
        "query": "Is the MS in Applied Data Science program STEM/OPT eligible?",
        "expected_answer": "The MS in Applied Data Science program is STEM/OPT eligible.",
    },
    {
        "query": "How many courses must you complete to earn UChicago's Master's in Applied Data Science?",
        "expected_answer": "To earn the MS-ADS degree students must successfully complete 12 courses (6 core, 4 elective, 2 Capstone) and our tailored Career Seminar.",
    },
]

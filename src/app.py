"""
Streamlit interface for the UChicago MS-ADS RAG assistant.

Run from the project root:

    streamlit run src/app.py
"""

from typing import Optional

import streamlit as st

try:
    from .rag import DsiRagAssistant
except ImportError:
    from rag import DsiRagAssistant

UCHICAGO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --maroon: #800000;
    --maroon-deep: #5C0000;
    --greystone: #D6D6CE;
    --dark-gray: #767676;
    --ink: #1A1A1A;
    --paper: #F4F4F1;
    --white: #FFFFFF;
}

html, body, [class*="css"] {
    font-family: "Source Sans 3", "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: var(--ink);
}

.stApp {
    background:
        radial-gradient(ellipse 120% 80% at 0% -10%, rgba(128, 0, 0, 0.08), transparent 55%),
        radial-gradient(ellipse 90% 60% at 100% 0%, rgba(214, 214, 206, 0.55), transparent 50%),
        linear-gradient(180deg, #F7F7F4 0%, #EFEFEA 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2A2A2A 0%, #1A1A1A 100%);
    border-right: 1px solid rgba(128, 0, 0, 0.35);
}

[data-testid="stSidebar"] * {
    color: #F4F4F1 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #D6D6CE !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(214, 214, 206, 0.25);
}

[data-testid="stSidebar"] .stSelectbox label {
    color: #D6D6CE !important;
}

.uc-hero {
    margin: -1rem -1rem 1.75rem -1rem;
    padding: 2.25rem 2rem 2rem 2rem;
    background:
        linear-gradient(135deg, #5C0000 0%, #800000 48%, #4A0000 100%);
    color: #FFFFFF;
    border-bottom: 4px solid #D6D6CE;
    position: relative;
    overflow: hidden;
}

.uc-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    opacity: 0.45;
}

.uc-hero-inner {
    position: relative;
    z-index: 1;
    max-width: 52rem;
}

.uc-eyebrow {
    font-family: "Source Sans 3", sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #D6D6CE;
    margin: 0 0 0.85rem 0;
}

.uc-brand {
    font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
    font-size: clamp(2rem, 4vw, 2.85rem);
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 0.65rem 0;
    color: #FFFFFF;
}

.uc-lede {
    font-family: "Source Sans 3", sans-serif;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.55;
    color: rgba(255, 255, 255, 0.88);
    margin: 0;
    max-width: 38rem;
}

.uc-section-label {
    font-family: "Source Sans 3", sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--maroon);
    margin: 0 0 0.75rem 0;
}

.uc-disclaimer {
    font-size: 0.88rem;
    color: var(--dark-gray);
    border-left: 3px solid var(--maroon);
    padding: 0.55rem 0 0.55rem 0.9rem;
    margin: 0 0 1.5rem 0;
    background: rgba(214, 214, 206, 0.35);
}

div.stButton > button {
    background: var(--white);
    color: var(--maroon);
    border: 1.5px solid var(--maroon);
    border-radius: 2px;
    font-family: "Source Sans 3", sans-serif;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: background 0.15s ease, color 0.15s ease;
}

div.stButton > button:hover {
    background: var(--maroon);
    color: var(--white);
    border-color: var(--maroon);
}

div.stButton > button[kind="secondary"] {
    border-color: var(--dark-gray);
    color: var(--ink);
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(214, 214, 206, 0.9);
    border-radius: 2px;
    padding: 0.35rem 0.75rem;
}

[data-testid="stChatInput"] textarea {
    font-family: "Source Sans 3", sans-serif;
}

.uc-footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--greystone);
    font-size: 0.8rem;
    color: var(--dark-gray);
    letter-spacing: 0.02em;
}

h1, h2, h3 {
    font-family: "Source Serif 4", Georgia, serif !important;
    color: var(--ink) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: "Source Serif 4", Georgia, serif !important;
    color: #FFFFFF !important;
}
</style>
"""


st.set_page_config(
    page_title="UChicago MS-ADS Assistant",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_assistant() -> DsiRagAssistant:
    """
    Load the RAG assistant once per Streamlit session.

    Returns
    -------
    DsiRagAssistant
        Cached assistant instance.
    """
    return DsiRagAssistant()


def run_question(question: str) -> None:
    """
    Send one question to the RAG assistant and display the result.

    Parameters
    ----------
    question : str
        User question text.
    """
    clean_question = question.strip()
    if not clean_question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the MS-ADS knowledge base..."):
            try:
                result = st.session_state.assistant.answer(clean_question)
                answer = result.get(
                    "answer",
                    "I could not generate an answer.",
                )
                st.markdown(answer)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )
            except Exception as error:
                st.error("An error occurred while generating the answer.")
                st.exception(error)


st.markdown(UCHICAGO_CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="uc-hero">
  <div class="uc-hero-inner">
    <p class="uc-eyebrow">The University of Chicago</p>
    <h1 class="uc-brand">MS in Applied Data Science</h1>
    <p class="uc-lede">
      Ask questions about admissions, curriculum, tuition, scholarships, deadlines, and program requirements.
    </p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="uc-disclaimer">
  Educational prototype. Verify admissions, tuition, deadline, visa, and
  policy details on the official UChicago website before acting on them.
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Data Science Institute")
    st.caption("MS-ADS Program Assistant")

    st.markdown("---")

    st.markdown("### Example Questions")
    example_questions = [
        "What are the core courses in the program?",
        "What are the admission requirements?",
        "How many courses are required to earn the degree?",
        "Does the online program provide visa sponsorship?",
        "What are the TOEFL and IELTS requirements?",
    ]
    selected_example = st.selectbox(
        "Choose an example",
        options=[""] + example_questions,
        format_func=lambda value: value or "Select a question",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(
        "Knowledge base drawn from datascience.uchicago.edu "
        "MS-ADS program pages."
    )


if "assistant" not in st.session_state:
    try:
        with st.spinner("Loading the MS-ADS knowledge base..."):
            st.session_state.assistant = load_assistant()
    except Exception as error:
        st.error(
            "The RAG system could not be loaded. Check the `.env` file, "
            "OpenAI API key, and FAISS vector-store files."
        )
        st.exception(error)
        st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown('<p class="uc-section-label">Quick Topics</p>', unsafe_allow_html=True)

button_columns = st.columns(4)
quick_questions = {
    "Curriculum": (
        "What are the core courses in the MS in Applied Data Science program?"
    ),
    "Admissions": (
        "What are the admission requirements for the MS in Applied Data Science program?"
    ),
    "Scholarships": "What scholarships are available for the program?",
    "International": (
        "What are the TOEFL and IELTS English language requirements?"
    ),
}

clicked_question: Optional[str] = None
for column, (label, question) in zip(button_columns, quick_questions.items()):
    with column:
        if st.button(label, use_container_width=True):
            clicked_question = question


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input(
    "Ask a question about the MS in Applied Data Science program"
)

question_to_run = None
if clicked_question:
    question_to_run = clicked_question
elif prompt:
    question_to_run = prompt
elif selected_example:
    question_to_run = selected_example

if question_to_run:
    run_question(question_to_run)

if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    """
<div class="uc-footer">
  Developed by Collin Kim & Somya Verma &middot; Powered by OpenAI and Streamlit
</div>
""",
    unsafe_allow_html=True,
)

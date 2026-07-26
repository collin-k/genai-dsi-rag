"""
Polished Streamlit interface for the UChicago MS-ADS RAG assistant.

Run from the project root:

    streamlit run src/app.py
"""

from typing import Dict, List

import streamlit as st

try:
    from .rag import DsiRagAssistant
except ImportError:
    from rag import DsiRagAssistant


st.set_page_config(
    page_title="UChicago MS-ADS RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_assistant() -> DsiRagAssistant:
    """Load the RAG assistant once per Streamlit session."""
    return DsiRagAssistant()


def render_sources(sources: List[Dict[str, str]]) -> None:
    """Display unique source links."""
    if not sources:
        st.info("No source links were returned.")
        return

    seen_urls = set()

    for source in sources:
        url = source.get("url", "").strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        page_title = source.get("page_title", "DSI webpage").strip()
        section_title = source.get("section_title", "").strip()

        label = (
            f"{page_title} — {section_title}"
            if section_title
            else page_title
        )

        st.markdown(f"- [{label}]({url})")


def run_question(question: str) -> None:
    """Send one question to the RAG assistant and display the result."""
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

                sources = result.get("sources", [])
                retrieval_results = result.get("retrieval_results", [])

                st.markdown(answer)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Retrieved Chunks",
                        len(retrieval_results),
                    )

                with col2:
                    st.metric(
                        "Dense Search",
                        "FAISS",
                    )

                with col3:
                    st.metric(
                        "Sparse Search",
                        "BM25",
                    )

                with st.expander("Sources"):
                    render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_count": len(retrieval_results),
                    }
                )

            except Exception as error:
                st.error(
                    "An error occurred while generating the answer."
                )
                st.exception(error)


st.title("🎓 UChicago MS-ADS RAG Chatbot")

st.subheader(
    "Ask questions about admissions, curriculum, tuition, scholarships, "
    "deadlines, and program requirements."
)

st.write(
    "This assistant uses hybrid retrieval to combine FAISS semantic search "
    "with BM25 keyword search before generating a grounded answer."
)

st.caption(
    "Educational prototype. Verify important admissions, tuition, deadline, "
    "visa, and policy information on the official UChicago website."
)


with st.sidebar:
    st.header("Project Information")

    st.markdown(
        """
**Course:** GEN AI Principles  
**Project:** RAG-based Interactive AI  
**Team:** Somya Verma and Team Member
"""
    )

    st.markdown("---")

    st.header("Example Questions")

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
    )

    st.markdown("---")

    st.header("Technology")

    st.markdown(
        """
- OpenAI GPT-4o-mini
- FAISS vector search
- BM25 keyword search
- Reciprocal Rank Fusion
- Streamlit
"""
    )


if "assistant" not in st.session_state:
    try:
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


st.markdown("### Quick Topics")

button_columns = st.columns(4)

quick_questions = {
    "📚 Curriculum": "What are the core courses in the MS in Applied Data Science program?",
    "🎓 Admissions": "What are the admission requirements for the MS in Applied Data Science program?",
    "💰 Scholarships": "What scholarships are available for the program?",
    "🌍 International": "What are the TOEFL and IELTS English language requirements?",
}

clicked_question = None

for column, (label, question) in zip(
    button_columns,
    quick_questions.items(),
):
    with column:
        if st.button(label, use_container_width=True):
            clicked_question = question


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "retrieval_count" in message:
                st.caption(
                    f"Retrieved {message['retrieval_count']} relevant chunks "
                    "using hybrid search."
                )

            if message.get("sources"):
                with st.expander("Sources"):
                    render_sources(message["sources"])


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


st.markdown("---")

st.caption(
    "Powered by OpenAI GPT-4o-mini, FAISS, BM25, Reciprocal Rank Fusion, "
    "and Streamlit."
)

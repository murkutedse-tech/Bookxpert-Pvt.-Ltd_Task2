"""Optional web UI (bonus). Run with:  streamlit run app.py"""

import streamlit as st

from rag_bot.config import Settings
from rag_bot.pipeline import QAEngine

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading vector store and embedding model ...")
def get_engine(provider: str) -> QAEngine:
    settings = Settings()
    settings.llm_provider = provider
    return QAEngine(settings)


st.title("📄 Document Q&A Bot")
st.caption("Retrieval-Augmented Generation over the documents in the data/ folder.")

with st.sidebar:
    st.header("Settings")
    default = Settings().llm_provider
    options = ["ollama", "gemini", "openai", "anthropic", "none"]
    provider = st.selectbox("LLM provider", options, index=options.index(default) if default in options else 0)
    top_k = st.slider("Top-k chunks", 1, 10, Settings().top_k)

try:
    engine = get_engine(provider)
except Exception as exc:  # index missing, model mismatch, ...
    st.error(str(exc))
    st.stop()

st.sidebar.markdown("**Indexed documents**")
for doc in engine.store.manifest["documents"]:
    st.sidebar.markdown(f"- {doc}")

question = st.text_input("Ask a question about the documents")
if question:
    with st.spinner("Retrieving and generating ..."):
        answer = engine.ask(question, top_k)
    (st.success if answer.grounded else st.warning)(answer.text)
    if answer.grounded:
        st.markdown("**Sources:** " + "; ".join(answer.citations))

    st.subheader("Retrieved chunks")
    used = {id(c) for c, _ in answer.context}
    for rank, (chunk, score) in enumerate(answer.retrieved, start=1):
        label = f"[{rank}] {chunk.citation} - score {score:.3f}"
        if id(chunk) not in used:
            label += " (below threshold, not sent to LLM)"
        with st.expander(label, expanded=rank == 1):
            st.write(chunk.text)

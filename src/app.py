import streamlit as st
from src.pipeline import RAGPipeline

st.set_page_config(page_title="DevDocs Intelligence", layout="wide")
st.title("DevDocs Intelligence")
st.caption("Hybrid documentation retrieval using BM25, FAISS, and Reciprocal Rank Fusion.")

@st.cache_resource(show_spinner="Loading local documents and embedding model...")
def get_pipeline():
    return RAGPipeline(documents_dir="data/raw")

try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Could not initialize the pipeline: {exc}")
    st.info("Check that data/raw contains .md or .txt files and that dependencies/model downloads are available.")
    st.stop()

question = st.text_input("Ask a technical question", placeholder="Why can a FastAPI request return HTTP 422?")
top_k = st.slider("Number of sources", 1, 8, 5)
use_ollama = st.checkbox("Use local Ollama generation (optional)", value=False)
ollama_model = st.text_input("Ollama model", value="llama3.2", disabled=not use_ollama)

if st.button("Search and answer", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Searching..."):
                result = pipeline.answer(question,top_k=5,use_ollama=True,ollama_model="llava:7b",)
            st.subheader("Answer / evidence")
            st.write(result["answer"])
            st.subheader("Sources")
            for i, source in enumerate(result["sources"], start=1):
                with st.expander(f"[{i}] {source['title']} — {source['source']}"):
                    st.write(f"RRF score: {source['score']:.5f}")
                    st.write(result["retrievals"][i-1]["text"])
        except Exception as exc:
            st.error(f"Search failed: {exc}")

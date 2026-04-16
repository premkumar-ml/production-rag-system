"""
app.py - Streamlit RAG Demo UI
Run: streamlit run app.py
"""
from __future__ import annotations
import os
import time
from pathlib import Path
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Production RAG System", page_icon="*", layout="wide")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(persist_dir="./chroma_store")

with st.sidebar:
    st.markdown("## RAG System")
    st.divider()
    st.markdown("### Ingest Documents")
    uploaded = st.file_uploader("Upload PDFs or Markdown files",
                                type=["pdf", "md", "txt"], accept_multiple_files=True)
    web_url = st.text_input("Or enter a URL to scrape")
    if st.button("Ingest", use_container_width=True):
        pipeline = load_pipeline()
        sources = []
        if uploaded:
            os.makedirs("./tmp_uploads", exist_ok=True)
            for f in uploaded:
                out_path = f"./tmp_uploads/{f.name}"
                with open(out_path, "wb") as out:
                    out.write(f.getbuffer())
                sources.append(out_path)
        if web_url.strip():
            sources.append(web_url.strip())
        if sources:
            with st.spinner("Ingesting..."):
                n = pipeline.ingest(sources)
            st.success(f"Done! {n} chunks indexed")
        else:
            st.warning("No sources provided.")
    st.divider()
    st.markdown("Built by Premkumar Narla")

st.markdown("# Production RAG System")
st.markdown("Hybrid retrieval - Cross-encoder reranking - Citation enforcement")
st.divider()

query = st.text_input("Ask a question about your documents",
                      placeholder="e.g. What are the recommended chunk sizes for RAG?")

if st.button("Ask") and query.strip():
    pipeline = load_pipeline()
    with st.spinner("Retrieving and generating..."):
        t0 = time.time()
        result = pipeline.query(query.strip())
        elapsed = round(time.time() - t0, 2)
    st.markdown("### Answer")
    st.markdown(f"Grounded: {result['grounded']} | {elapsed}s | {result['num_chunks_retrieved']} chunks")
    st.info(result["answer"])
    if result["sources"]:
        st.markdown("### Sources")
        for src in result["sources"]:
            st.markdown(f"**[{src['chunk_id']}]** {Path(src['source']).name} | score: {src['score']}")
            st.caption(src["preview"])

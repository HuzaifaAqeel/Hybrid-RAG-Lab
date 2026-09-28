"""Hybrid RAG Lab -- Streamlit UI comparing naive vs hybrid retrieval side by side.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag import RAGLab  # noqa: E402

ROOT = Path(__file__).resolve().parent
DB_PATH = str(ROOT / "hybrid_rag.db")

st.set_page_config(page_title="Hybrid RAG Lab", page_icon="🔬", layout="wide")
st.title("🔬 Hybrid RAG Lab")
st.caption("Naive vector search vs hybrid BM25 + vector search with cross-encoder reranking — "
           "on embedded Milvus (milvus-lite).")


@st.cache_resource(show_spinner="Indexing corpus into Milvus…")
def get_lab(with_reranker: bool, first_stage_k: int) -> RAGLab:
    return RAGLab(str(ROOT / "data" / "corpus.json"), db_path=DB_PATH,
                  with_reranker=with_reranker, first_stage_k=first_stage_k)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k results", 1, 10, 5)
    first_stage_k = st.slider("First-stage candidates", 5, 18, 10)
    use_reranker = st.toggle("Cross-encoder rerank", value=True)
    generate = st.toggle("Generate answer (needs OPENAI_API_KEY)", value=False)
    st.divider()
    st.caption(f"Corpus: 18 hand-written docs on RAG · Milvus file: `{DB_PATH}`")

lab = get_lab(use_reranker, first_stage_k)
lab.retriever.rrf_k = 60

with st.expander("📚 Corpus", expanded=False):
    for doc_id in lab.ids:
        st.markdown(f"**{lab.titles[doc_id]}** `{doc_id}`")
        st.caption(lab.retriever._id_to_text[doc_id][:220] + "…")

query = st.text_input("Ask a question", value="How do I combine keyword and vector search?")
if st.button("Search", type="primary") and query.strip():
    with st.spinner("Retrieving…"):
        result = lab.compare(query, top_k=top_k)

    naive_ids = {h.doc_id for h in result.naive_hits}
    hybrid_ids = {h.doc_id for h in result.hybrid_hits}
    overlap = len(naive_ids & hybrid_ids)
    st.info(f"⏱️ naive: {result.naive_ms:.0f} ms · hybrid: {result.hybrid_ms:.0f} ms · "
            f"overlap: {overlap}/{top_k} documents")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Naive — vector only")
        for rank, hit in enumerate(result.naive_hits, 1):
            st.markdown(f"**{rank}. {lab.titles[hit.doc_id]}** `{hit.doc_id}` "
                        f"(cos-sim {hit.score:.3f})")
            st.caption(hit.text[:280] + "…")
    with col2:
        st.subheader("Hybrid — BM25 + vector → RRF → rerank")
        for rank, hit in enumerate(result.hybrid_hits, 1):
            rerank = hit.detail.get("rerank_score")
            extra = f"rerank {rerank:.3f}" if rerank is not None else f"rrf {hit.score:.4f}"
            st.markdown(f"**{rank}. {lab.titles[hit.doc_id]}** `{hit.doc_id}` ({extra})")
            st.caption(hit.text[:280] + "…")

    if generate:
        with st.spinner("Generating grounded answer…"):
            answer = lab.answer(query, top_k=top_k).answer
        st.subheader("💬 Grounded answer")
        st.markdown(answer or "*empty*")
else:
    st.info("Type a question above and hit **Search** — e.g. "
            "“What is reciprocal rank fusion?” or “How do I evaluate retrieval?”.")

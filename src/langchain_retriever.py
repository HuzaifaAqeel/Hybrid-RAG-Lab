"""LangChain integration: expose :class:`HybridRetriever` as a LangChain retriever.

This lets the lab's hybrid pipeline (BM25 + dense vectors + RRF + rerank)
drop into any LangChain chain, agent, or ``RetrievalQA``-style workflow::

    from src.langchain_retriever import LangChainHybridRetriever

    retriever = LangChainHybridRetriever(hybrid_retriever, top_k=5, mode="hybrid")
    docs = retriever.invoke("What is hybrid search?")
"""

from __future__ import annotations

from typing import Literal

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .retrieval import HybridRetriever


class LangChainHybridRetriever(BaseRetriever):
    """A ``BaseRetriever`` adapter around :class:`HybridRetriever`.

    Parameters
    ----------
    retriever:
        A fully constructed :class:`~src.retrieval.HybridRetriever`.
    top_k:
        Number of documents to return per query.
    mode:
        ``"hybrid"`` (BM25 + vectors + RRF + rerank) or ``"naive"``
        (dense vectors only) -- handy for side-by-side comparisons
        inside LangChain pipelines.
    """

    retriever: HybridRetriever
    top_k: int = 5
    mode: Literal["hybrid", "naive"] = "hybrid"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:  # noqa: D102
        if self.mode == "naive":
            hits, latency_ms = self.retriever.naive(query, top_k=self.top_k)
        else:
            hits, latency_ms = self.retriever.hybrid(query, top_k=self.top_k)
        return [
            Document(
                page_content=hit.text,
                metadata={
                    "doc_id": hit.doc_id,
                    "score": hit.score,
                    "stage": hit.detail.get("stage"),
                    "latency_ms": latency_ms,
                },
            )
            for hit in hits
        ]

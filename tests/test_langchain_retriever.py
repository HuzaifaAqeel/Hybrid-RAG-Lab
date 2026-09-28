"""Tests for the LangChain retriever adapter (stubbed embedder/store, no models)."""

from src.langchain_retriever import LangChainHybridRetriever
from src.retrieval import HybridRetriever


class StubEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


class StubStore:
    def __init__(self, ids):
        self._ids = ids

    def vector_search(self, qvec, top_k=5):
        return [(i, 0.9) for i in self._ids[:top_k]]


def _hybrid():
    ids = ["d1", "d2", "d3"]
    texts = ["milvus vector database", "bm25 keyword search", "hybrid retrieval rerank"]
    return HybridRetriever(ids, texts, StubEmbedder(), StubStore(ids), reranker=None)


def test_langchain_retriever_returns_documents():
    lc = LangChainHybridRetriever(retriever=_hybrid(), top_k=2, mode="hybrid")
    docs = lc.invoke("vector search")
    assert len(docs) == 2
    assert all(hasattr(d, "page_content") for d in docs)
    assert docs[0].metadata["doc_id"] in {"d1", "d2", "d3"}
    assert docs[0].metadata["stage"] == "rrf"


def test_langchain_retriever_naive_mode():
    lc = LangChainHybridRetriever(retriever=_hybrid(), top_k=1, mode="naive")
    docs = lc.invoke("vector search")
    assert len(docs) == 1
    assert docs[0].metadata["stage"] == "vector"

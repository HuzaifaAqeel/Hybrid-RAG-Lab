"""RAG lab: index a corpus, compare naive vs hybrid retrieval, answer with an LLM.

The LLM step is optional: set ``OPENAI_API_KEY`` (and optionally
``OPENAI_BASE_URL`` for OpenAI-compatible endpoints, or ``LLM_MODEL`` to pick
the model) to generate grounded answers. Without a key, ``answer()`` returns
the retrieved context so the retrieval comparison still works offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .retrieval import Embedder, Hit, HybridRetriever, Reranker
from .store import VectorStore


@dataclass
class LabResult:
    query: str
    naive_hits: list[Hit]
    hybrid_hits: list[Hit]
    naive_ms: float
    hybrid_ms: float
    answer: str | None = None


class RAGLab:
    def __init__(self, corpus_path: str, db_path: str = "./hybrid_rag.db",
                 with_reranker: bool = True, first_stage_k: int = 10):
        with open(corpus_path, encoding="utf-8") as fh:
            corpus = json.load(fh)
        self.ids = [d["id"] for d in corpus]
        self.texts = [d["text"] for d in corpus]
        self.titles = {d["id"]: d.get("title", d["id"]) for d in corpus}
        self.first_stage_k = first_stage_k

        self.embedder = Embedder()
        vectors = self.embedder.encode(self.texts)
        self.store = VectorStore(db_path=db_path, dim=self.embedder.dim)
        self.store.insert(self.ids, self.texts, vectors)
        self.retriever = HybridRetriever(
            self.ids, self.texts, self.embedder, self.store,
            reranker=Reranker() if with_reranker else None,
        )

    def compare(self, query: str, top_k: int = 5) -> LabResult:
        naive_hits, naive_ms = self.retriever.naive(query, top_k=top_k)
        hybrid_hits, hybrid_ms = self.retriever.hybrid(
            query, top_k=top_k, first_stage_k=self.first_stage_k)
        return LabResult(query=query, naive_hits=naive_hits,
                         hybrid_hits=hybrid_hits, naive_ms=naive_ms,
                         hybrid_ms=hybrid_ms)

    def answer(self, query: str, top_k: int = 5) -> LabResult:
        result = self.compare(query, top_k=top_k)
        result.answer = self._generate(query, result.hybrid_hits)
        return result

    def _generate(self, query: str, hits: list[Hit]) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            context = "\n\n".join(
                f"[{h.doc_id}] {h.text}" for h in hits
            )
            return ("(No OPENAI_API_KEY set -- returning retrieved context only.)\n\n"
                    + context)
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        context = "\n\n".join(f"[{h.doc_id}] {h.text}" for h in hits)
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.0,
            messages=[
                {"role": "system",
                 "content": "Answer the question using only the provided context. "
                            "Cite sources like [doc-01]. If the context lacks the "
                            "answer, say so."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return response.choices[0].message.content or ""

    def close(self) -> None:
        self.store.close()

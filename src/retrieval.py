"""Hybrid retrieval: BM25 + dense vectors fused with RRF, then reranked.

Pipelines:
  - ``naive``: dense vector search only (the baseline every RAG demo ships).
  - ``hybrid``: BM25 and vector search fused with reciprocal rank fusion,
    then the top candidates are reranked with a cross-encoder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .bm25 import BM25
from .store import VectorStore


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    detail: dict = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id-lists: score(id) = sum(1 / (k + rank)). Best first."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class Embedder:
    """Lazy sentence-transformers wrapper (downloads model on first use)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        load = self._load()
        if hasattr(load, "get_embedding_dimension"):
            return load.get_embedding_dimension()
        return load.get_sentence_embedding_dimension()  # older versions

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self._load().encode(texts, normalize_embeddings=True).tolist()


class Reranker:
    """Lazy cross-encoder wrapper."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._load().predict([(query, h.text) for h in hits]).tolist()
        ordered = sorted(zip(hits, scores), key=lambda p: p[1], reverse=True)
        out = []
        for hit, score in ordered[:top_k]:
            hit.detail["rerank_score"] = float(score)
            out.append(hit)
        return out


class HybridRetriever:
    def __init__(self, ids: list[str], texts: list[str], embedder: Embedder,
                 store: VectorStore, rrf_k: int = 60, reranker: Reranker | None = None):
        self.ids = ids
        self.texts = texts
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k
        self.reranker = reranker
        self.bm25 = BM25(texts)
        self._id_to_text = dict(zip(ids, texts))

    def naive(self, query: str, top_k: int = 5) -> tuple[list[Hit], float]:
        """Dense vector search only."""
        t0 = time.perf_counter()
        qvec = self.embedder.encode([query])[0]
        results = self.store.vector_search(qvec, top_k=top_k)
        hits = [
            Hit(doc_id=i, text=self._id_to_text[i], score=s,
                detail={"stage": "vector"})
            for i, s in results
        ]
        return hits, (time.perf_counter() - t0) * 1000

    def hybrid(self, query: str, top_k: int = 5,
               first_stage_k: int = 10) -> tuple[list[Hit], float]:
        """BM25 + vector search -> RRF fusion -> cross-encoder rerank."""
        t0 = time.perf_counter()
        qvec = self.embedder.encode([query])[0]
        vec_ranking = [doc_id for doc_id, _ in self.store.vector_search(qvec, top_k=first_stage_k)]
        bm25_ranking = [self.ids[i] for i, _ in self.bm25.ranked(query, top_k=first_stage_k)]
        fused = reciprocal_rank_fusion([bm25_ranking, vec_ranking], k=self.rrf_k)
        candidates = [
            Hit(doc_id=doc_id, text=self._id_to_text[doc_id], score=score,
                detail={"stage": "rrf"})
            for doc_id, score in fused[:first_stage_k]
        ]
        if self.reranker is not None:
            hits = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            hits = candidates[:top_k]
        return hits, (time.perf_counter() - t0) * 1000

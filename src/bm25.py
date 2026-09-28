"""Pure-Python BM25 sparse retrieval.

Implements the BM25 ranking function (Robertson & Zaragoza) with
tokenization, inverse document frequency with the standard +0.5 smoothing,
term-frequency saturation (k1) and document-length normalization (b).
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; drops punctuation and stopword-free."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Corpus-level BM25 scorer. Build once, score many queries."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized: list[list[str]] = [tokenize(d) for d in documents]
        self.doc_lengths = [len(t) for t in self.tokenized]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

        df: Counter[str] = Counter()
        for tokens in self.tokenized:
            df.update(set(tokens))
        n = len(documents)
        # IDF with +0.5 smoothing; floored at 0 to ignore ultra-common terms.
        self.idf = {
            term: max(0.0, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of every document for the query (order = corpus order)."""
        qterms = tokenize(query)
        out = [0.0] * len(self.documents)
        for doc_id, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            dl = self.doc_lengths[doc_id] or 1
            score = 0.0
            for term in set(qterms):
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator
            out[doc_id] = score
        return out

    def ranked(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """(doc_index, score) sorted best-first, top_k only."""
        scored = self.scores(query)
        order = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
        return [(i, scored[i]) for i in order[:top_k]]

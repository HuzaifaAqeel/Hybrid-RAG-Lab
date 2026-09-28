#!/usr/bin/env python
"""CLI experiment: naive vs hybrid retrieval on a few probe queries.

Runs entirely offline (no LLM key needed) and prints a side-by-side
comparison: top doc, latency, and ranking overlap per query.

Usage:
    python experiments/compare.py [--top-k 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import RAGLab  # noqa: E402

PROBES = [
    "How do I combine keyword and vector search?",
    "What is reciprocal rank fusion?",
    "Which index is best for approximate nearest neighbor search?",
    "How do I evaluate whether my retrieval is good?",
    "What are the risks of letting an LLM read retrieved documents?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    lab = RAGLab(str(root / "data" / "corpus.json"),
                 db_path=str(root / "hybrid_rag.db"))
    try:
        print(f"{'query':55s} | {'naive top-1':10s} | {'hybrid top-1':10s} | overlap")
        print("-" * 100)
        for query in PROBES:
            result = lab.compare(query, top_k=args.top_k)
            naive_ids = [h.doc_id for h in result.naive_hits]
            hybrid_ids = [h.doc_id for h in result.hybrid_hits]
            overlap = len(set(naive_ids) & set(hybrid_ids))
            print(f"{query[:55]:55s} | {naive_ids[0]:10s} | {hybrid_ids[0]:10s} "
                  f"| {overlap}/{args.top_k}  "
                  f"({result.naive_ms:.0f}ms vs {result.hybrid_ms:.0f}ms)")
    finally:
        lab.close()


if __name__ == "__main__":
    main()

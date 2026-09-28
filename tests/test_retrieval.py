"""Unit tests for the pure-Python BM25 implementation and RRF fusion.

No models, no network -- these run anywhere in milliseconds.
"""

from src.bm25 import BM25, tokenize
from src.retrieval import reciprocal_rank_fusion


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Hello, World! 123") == ["hello", "world", "123"]


def test_exact_term_match_outranks_unrelated_doc():
    bm25 = BM25(["the cat sat on the mat", "quantum field theory"])
    top = bm25.ranked("quantum theory", top_k=2)
    assert top[0][0] == 1
    assert top[0][1] > top[1][1]


def test_repeated_rare_term_scores_higher():
    bm25 = BM25(["milvus vector database", "milvus milvus milvus vector"])
    top = bm25.ranked("milvus", top_k=2)
    assert top[0][0] == 1


def test_scores_length_matches_corpus():
    docs = ["a b c", "d e f", "g h i"]
    assert len(BM25(docs).scores("a b")) == 3


def test_rrf_rewards_documents_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] in {"a", "b"}  # appearing in both lists beats single-list docs
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_k_dampens_top_ranks():
    # doc "a" appears in both lists, "b" in only one
    strict = reciprocal_rank_fusion([["a", "b"], ["a"]], k=1)
    loose = reciprocal_rank_fusion([["a", "b"], ["a"]], k=60)
    # with k=1 the doubly-supported doc dominates; with k=60 scores are closer
    gap_strict = strict[0][1] - strict[1][1]
    gap_loose = loose[0][1] - loose[1][1]
    assert strict[0][0] == "a" and loose[0][0] == "a"
    assert gap_strict > gap_loose

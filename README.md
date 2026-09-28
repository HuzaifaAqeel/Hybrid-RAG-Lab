# Hybrid RAG Lab

A hands-on lab for **hybrid retrieval**: BM25 keyword search + dense vector search
over embedded **Milvus** (milvus-lite — no server, no Docker), fused with
**reciprocal rank fusion** and reranked by a **cross-encoder**. A Streamlit UI
puts **naive vector-only retrieval** and the **hybrid pipeline** side by side so
you can see exactly where hybrid wins.

![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)

## The idea

Naive RAG does one thing: embed the query, cosine-search the corpus, generate.
It misses exact terms ("RRF", "HNSW") that a keyword index catches instantly,
while pure keyword search misses paraphrases that embeddings catch. The hybrid
pipeline runs both, fuses the rankings with reciprocal rank fusion, and lets a
cross-encoder — which reads the query and document *together* — pick the final
order.

```
query ─┬─▶ BM25 (sparse) ────────┐
       │                          ├─▶ RRF fusion ─▶ cross-encoder rerank ─▶ top-k
       └─▶ Milvus dense search ───┘
```

## Quickstart

```bash
git clone https://github.com/HuzaifaAqeel/Hybrid-RAG-Lab.git
cd Hybrid-RAG-Lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads two small models from Hugging Face
(`all-MiniLM-L6-v2` for embeddings, `ms-marco-MiniLM-L-6-v2` for reranking).
Everything else — including Milvus — runs locally.

### 1. Compare pipelines from the terminal

```bash
python experiments/compare.py --top-k 3
```

Real output (MiniLM embeddings + cross-encoder rerank, milvus-lite):

```
query                                                   | naive top-1 | hybrid top-1 | overlap
----------------------------------------------------------------------------------------------------
How do I combine keyword and vector search?             | doc-04      | doc-04       | 2/3  (17ms vs 680ms)
What is reciprocal rank fusion?                         | doc-05      | doc-05       | 2/3  (44ms vs 340ms)
Which index is best for approximate nearest neighbor se | doc-08      | doc-08       | 2/3  (14ms vs 460ms)
How do I evaluate whether my retrieval is good?         | doc-11      | doc-11       | 2/3  (15ms vs 502ms)
What are the risks of letting an LLM read retrieved doc | doc-16      | doc-16       | 2/3  (16ms vs 367ms)
```

Top-1 agrees on these probes, but the #2/#3 slots move — try the query
`HNSW` in the UI: naive vector search ranks the LangChain doc second, while
the hybrid pipeline surfaces the Milvus and BM25 docs instead.

### 2. Side-by-side UI

```bash
streamlit run app.py
```

Type a question and watch both pipelines rank the corpus next to each other,
with latency, cosine scores, RRF scores and rerank scores.

### 3. Use it from LangChain

The hybrid pipeline also ships as a LangChain retriever, so it drops into
any chain or agent:

```python
from src.rag import RAGLab
from src.langchain_retriever import LangChainHybridRetriever

lab = RAGLab("data/corpus.json", with_reranker=True)
retriever = LangChainHybridRetriever(retriever=lab.retriever, top_k=5, mode="hybrid")
docs = retriever.invoke("What is reciprocal rank fusion?")
```

`mode="naive"` switches the same adapter to vector-only retrieval — handy
for A/B comparisons inside LangChain pipelines.

### 4. Grounded answers (optional)

Set `OPENAI_API_KEY` (or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`)
and toggle **Generate answer** in the UI — answers are generated strictly from
the retrieved hybrid context with `[doc-id]` citations.

## Project structure

```
Hybrid-RAG-Lab/
├── app.py                 # Streamlit side-by-side comparison UI
├── requirements.txt
├── .env.example
├── data/
│   └── corpus.json        # 18 hand-written docs on RAG/retrieval
├── src/
│   ├── bm25.py            # pure-Python BM25 (tokenize, IDF, k1/b saturation)
│   ├── store.py           # embedded Milvus (milvus-lite) vector store
│   ├── retrieval.py       # RRF fusion, embedder, cross-encoder, HybridRetriever
│   ├── langchain_retriever.py  # LangChain BaseRetriever adapter (hybrid/naive)
│   └── rag.py             # RAGLab: index -> compare -> optional grounded answer
├── experiments/
│   └── compare.py         # CLI: naive vs hybrid over probe queries
└── tests/
    ├── test_retrieval.py          # offline unit tests (BM25 + RRF)
    └── test_langchain_retriever.py  # adapter tests (stubbed, no models)
```

## What to try

- Ask *"What is reciprocal rank fusion?"* — both pipelines agree (exact term).
- Ask *"HNSW"* — naive vector search puts the LangChain doc second, while
  hybrid surfaces the Milvus and BM25 docs instead.
- Ask *"Which index is best for approximate nearest neighbor search?"* —
  both pipelines agree on the HNSW doc, but compare their #2/#3 picks.
- Toggle the reranker off and see how much the final order moves.
- Add your own documents to `data/corpus.json` and re-run.

## License

MIT — see [LICENSE](LICENSE).

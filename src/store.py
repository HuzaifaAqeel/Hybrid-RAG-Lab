"""Embedded Milvus vector store backed by milvus-lite.

Uses the native ``milvus_lite`` embedded engine, so the whole vector
database lives in a single directory -- no server, no Docker. Documents
are stored with their text plus a dense embedding; search is brute-force
cosine over the stored vectors (exact for lab-scale corpora).

Note: for ``COSINE`` searches milvus-lite returns the cosine *similarity*
in the ``distance`` field (1.0 = identical, higher = better), so we use it
directly as the score.
"""

from __future__ import annotations

from milvus_lite import CollectionSchema, DataType, FieldSchema, MilvusLite

COLLECTION = "hybrid_rag_docs"
ID_FIELD = "doc_id"
TEXT_FIELD = "text"
VECTOR_FIELD = "dense"


class VectorStore:
    def __init__(self, db_path: str = "./hybrid_rag.db", dim: int = 384,
                 collection: str = COLLECTION):
        self.db_path = db_path
        self.dim = dim
        self.collection = collection
        self.db = MilvusLite(db_path)
        if collection in self.db.list_collections():
            self.db.drop_collection(collection)
        schema = CollectionSchema(fields=[
            FieldSchema(name=ID_FIELD, dtype=DataType.VARCHAR,
                        is_primary=True, max_length=64),
            FieldSchema(name=TEXT_FIELD, dtype=DataType.VARCHAR,
                        max_length=65535),
            FieldSchema(name=VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR,
                        dim=dim),
        ])
        self.col = self.db.create_collection(collection, schema)

    def insert(self, ids: list[str], texts: list[str],
               vectors: list[list[float]]) -> int:
        rows = [
            {ID_FIELD: i, TEXT_FIELD: t, VECTOR_FIELD: v}
            for i, t, v in zip(ids, texts, vectors)
        ]
        return len(self.col.insert(rows))

    def vector_search(self, query_vector: list[float],
                      top_k: int = 5) -> list[tuple[str, float]]:
        """Returns (doc_id, cosine similarity) best-first."""
        hits = self.col.search(
            [query_vector],
            top_k=top_k,
            metric_type="COSINE",
            output_fields=[TEXT_FIELD],
        )[0]
        return [(h["id"], float(h["distance"])) for h in hits]

    def fetch_text(self, doc_id: str) -> str:
        rows = self.col.get([doc_id], output_fields=[TEXT_FIELD])
        return rows[0].get(TEXT_FIELD, "") if rows else ""

    def close(self) -> None:
        self.db.close()

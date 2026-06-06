from typing import List, Dict
from collections import defaultdict

from rag_core.llm.base import ModelClient
from rag_core.vectorstore.faiss_store import FaissStore

import whoosh.index as whoosh_index
from whoosh.qparser import MultifieldParser


class HybridRetriever:

    def __init__(
        self,
        model_client: ModelClient,
        faiss_store: FaissStore,
        whoosh_index_dir: str
    ):
        self.model = model_client
        self.faiss = faiss_store
        self.whoosh_index_dir = whoosh_index_dir

    # -----------------------------
    # Semantic Search
    # -------------------------------
    def semantic_search(self, query: str, k: int = 10):

        q_vec = self.model.embed([query])

        if hasattr(q_vec, "reshape"):
            q_vec = q_vec.reshape(1, -1)

        results = self.faiss.search(q_vec, k=k)

        return results

    # -----------------------------
    # Keyword Search (exact / partial matches)
    # -----------------------------
    def keyword_search(self, query: str, k: int = 10):

        idx = whoosh_index.open_dir(self.whoosh_index_dir)

    
        qp = MultifieldParser(["content"], schema=idx.schema)

        q = qp.parse(query)

        with idx.searcher() as s:

            hits = s.search(q, limit=k)

            return [
                {
                    "source": h["source"],
                    "text": h["content"]
                }
                for h in hits
            ]

    # -----------------------------
    # Reciprocal Rank Fusion (RRF)
    # -----------------------------
    def reciprocal_rank_fusion(
        self,
        rankings: List[List[Dict]],
        k: int = 60
    ):

        scores = defaultdict(float)
        documents = {}

        for ranking in rankings:

            for rank, doc in enumerate(ranking):

                doc_id = (
                    doc.get("source"),
                    doc.get("text")[:100]
                )

                documents[doc_id] = doc

                scores[doc_id] += 1 / (k + rank + 1)

        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            documents[doc_id]
            for doc_id, _ in sorted_docs
        ]

    # -----------------------------
    # Hybrid Search
    # -----------------------------
    def hybrid(self, query: str, k: int = 5):

        semantic_results = self.semantic_search(query, k=k)

        keyword_results = self.keyword_search(query, k=k)

        fused = self.reciprocal_rank_fusion([
            semantic_results,
            keyword_results
        ])

        return fused[:k]

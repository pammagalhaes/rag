from typing import List, Dict
from collections import defaultdict
from rag_core.llm.base import ModelClient
from rag_core.vectorstore.faiss_store import FaissStore
from whoosh.qparser import MultifieldParser
import whoosh.index as whoosh_index


class HybridRetriever:
    def __init__(self, model_client: ModelClient, faiss_store: FaissStore, whoosh_index_dir: str):
        self.model = model_client
        self.faiss = faiss_store
        self.whoosh_index_dir = whoosh_index_dir


    def semantic_search(self, query: str, k: int = 5):
        q_vec = self.model.embed([query])
        if hasattr(q_vec, "reshape"):
            q_vec = q_vec.reshape(1, -1)
        return self.faiss.search(q_vec, k=k)


    def keyword_search(self, query: str, k: int = 5):
        idx = whoosh_index.open_dir(self.whoosh_index_dir)
        qp = MultifieldParser(["content"], schema=idx.schema)
        q = qp.parse(query)
        with idx.searcher() as s:
            hits = s.search(q, limit=k)
            results = []
            for h in hits:
                doc = {
                    "source": h["source"],
                    "text": h["content"],
                    "score": float(h.score) if h.score is not None else None,
                }
                # Forward page / slide metadata when the schema has those fields.
                if "page" in idx.schema.names():
                    raw_page = h.get("page", None)
                    if raw_page is not None and raw_page != -1:
                        doc["page"] = int(raw_page)
                if "slide" in idx.schema.names():
                    raw_slide = h.get("slide", None)
                    if raw_slide is not None and raw_slide != -1:
                        doc["slide"] = int(raw_slide)
                results.append(doc)
            return results

    def reciprocal_rank_fusion(self, rankings: List[List[Dict]], k: int = 60):
        scores = defaultdict(float)
        documents = {}

        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                doc_id = (doc.get("source"), doc.get("text")[:100])
                documents[doc_id] = doc
                scores[doc_id] += 1 / (k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [documents[doc_id] for doc_id, _ in sorted_docs]

    def hybrid(self, query: str, k: int = 5):
        sem = self.semantic_search(query, k=k)
        kw = self.keyword_search(query, k=k)
        fused = self.reciprocal_rank_fusion([sem, kw])
        return fused[:k]


from typing import List
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
            return [{"source": h["source"], "text": h["content"]} for h in hits]


    def hybrid(self, query: str, k: int = 5):
        sem = self.semantic_search(query, k=k)
        kw = self.keyword_search(query, k=k)

        seen = set()
        merged = []

        for r in sem + kw:
            key = (r.get("source"), r.get("text")[:80])

            if key not in seen:
                merged.append(r)
                seen.add(key)

            if len(merged) >= k:
                break

        return merged

// TEST CHANGE

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict

class FaissStore:
    def __init__(self, dim: int, index_path: str, meta_path: str):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            # If the saved index dimension doesn't match the model dimension, recreate it
            if self.index.d != self.dim:
                print(f"[FAISS WARNING] Index dimension ({self.index.d}) ≠ model dimension ({self.dim}).")
                print("[FAISS] Recreating empty index with correct dimension.")
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadatas = []
                return

            with open(self.meta_path, "rb") as f:
                self.metadatas = pickle.load(f)

            if self.index.ntotal != len(self.metadatas):
                raise ValueError(
                    "FAISS index and metadata are out of sync: "
                    f"index contains {self.index.ntotal} vectors, but "
                    f"metadata contains {len(self.metadatas)} entries. "
                    "Rebuild both files together."
                )
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadatas = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        vectors = np.asarray(vectors, dtype=np.float32).copy()
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Incorrect dimension: FAISS expects {self.index.d}, "
                f"but vectors have {vectors.shape[1]}"
            )

        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def search(self, query_vec: np.ndarray, k: int = 5):
        query_vec = np.asarray(query_vec, dtype=np.float32).copy()
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.metadatas):
                meta = self.metadatas[idx].copy()

                result = {
                    "text": meta.get("text", meta.get("page_content", "")),
                    "source": meta.get("source", ""),
                    "page": meta.get("page", None),
                    "chunk_id": meta.get("chunk_id", None),
                    "backend": "faiss",
                    "score": float(score)
                }
                results.append(result)
        return results


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
            with open(self.meta_path, "rb") as f:
                self.metadatas = pickle.load(f)

        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadatas = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict]):
        self.index.add(vectors)
        self.metadatas.extend(metadatas)

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def search(self, query_vec: np.ndarray, k: int = 5):
        D, I = self.index.search(query_vec, k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadatas):
                results.append(self.metadatas[idx])
        return results

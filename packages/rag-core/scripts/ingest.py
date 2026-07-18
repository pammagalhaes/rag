import os
import glob
import math
from typing import List

from dotenv import load_dotenv

load_dotenv()

from rag_core.ingestion.loaders import load_file
from rag_core.llm.transformers_client import TransformersClient
from rag_core.vectorstore.faiss_store import FaissStore

from whoosh import index as whoosh_index
from whoosh.fields import Schema, TEXT, ID, NUMERIC


def ensure_whoosh_index(whoosh_dir: str):
    if not os.path.exists(whoosh_dir):
        os.makedirs(whoosh_dir, exist_ok=True)

    if not whoosh_index.exists_in(whoosh_dir):
        schema = Schema(content=TEXT(stored=True), source=ID(stored=True), page=NUMERIC(stored=True))
        whoosh_index.create_in(whoosh_dir, schema)


def ingest_dir(
    source_dir: str,
    faiss_index_path: str,
    faiss_meta_path: str,
    whoosh_dir: str,
    embed_batch: int = 32,
    dim: int = 1536,
):
    model = TransformersClient()

    store = FaissStore(dim=dim, index_path=faiss_index_path, meta_path=faiss_meta_path)

    ensure_whoosh_index(whoosh_dir)
    idx = whoosh_index.open_dir(whoosh_dir)

    files = []
    for ext in ("*.pdf", "*.txt", "*.md", "*.pptx"):
        files.extend(glob.glob(os.path.join(source_dir, ext)))

    if not files:
        print(f"No files found in {source_dir}")
        return

    all_docs = []
    for f in files:
        try:
            docs = load_file(f)
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} docs from {f}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")

    # Build Whoosh index and collect texts for embeddings
    writer = idx.writer()
    texts = []
    metadatas = []

    for doc in all_docs:
        text = doc.page_content.strip()
        if not text:
            continue

        src = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)

        # add to whoosh (guard against existing index schema differences)
        if "page" in idx.schema.names():
            writer.add_document(content=text, source=src, page=page if page is not None else -1)
        else:
            writer.add_document(content=text, source=src)

        texts.append(text)
        metadatas.append({
            "page_content": text,
            "source": src,
            "page": page,
        })

    writer.commit()
    print(f"Indexed {len(texts)} documents into Whoosh at {whoosh_dir}")

    # Create embeddings in batches and add to FAISS
    total = len(texts)
    for i in range(0, total, embed_batch):
        batch_texts = texts[i : i + embed_batch]
        batch_meta = metadatas[i : i + embed_batch]
        try:
            vecs = model.embed(batch_texts)
        except Exception as e:
            print(f"Embedding failed for batch starting at {i}: {e}")
            raise

        # ensure numpy 2D
        import numpy as np

        arr = np.array(vecs)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        store.add(arr, batch_meta)
        print(f"Added batch {i}-{i+len(batch_texts)} to FAISS")

    store.persist()
    print(f"FAISS index persisted to {faiss_index_path} (meta: {faiss_meta_path})")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="data/raw", help="Directory with source files")
    p.add_argument("--faiss-index", default="data/index/faiss.index", help="FAISS index path")
    p.add_argument("--faiss-meta", default="data/index/faiss_meta.pkl", help="FAISS metadata path")
    p.add_argument("--whoosh-dir", default="data/index/whoosh", help="Whoosh index dir")
    p.add_argument("--batch", type=int, default=32, help="Embedding batch size")
    args = p.parse_args()

    ingest_dir(
        source_dir=args.source_dir,
        faiss_index_path=args.faiss_index,
        faiss_meta_path=args.faiss_meta,
        whoosh_dir=args.whoosh_dir,
        embed_batch=args.batch,
    )

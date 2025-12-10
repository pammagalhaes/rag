from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.configs.loader import load_config
from src.utils.logging_setup import init_logging

cfg = load_config("default.yaml")
init_logging("logging.yaml")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request, exc):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

class Query(BaseModel):
    question: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask")
@limiter.limit("5/minute")
async def ask(request: Request, q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="question missing")
    # Lazy init to avoid heavy import at startup
    from src.llm.transformers_client import TransformersClient
    from src.vectorstore.faiss_store import FaissStore
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.agents.agent_runner import AgentRunner

    model = TransformersClient()
    faiss = FaissStore(dim=384, index_path=cfg['vectorstore']['faiss_index_path'], meta_path=cfg['vectorstore']['faiss_meta_path'])
    retriever = HybridRetriever(model_client=model, faiss_store=faiss, whoosh_index_dir="data/index/whoosh")
    agent = AgentRunner(model_client=model, retriever=retriever)
    res = agent.run(q.question)
    return {"answer": res}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    import os
    from src.ingestion.loaders import load_file
    from src.ingestion.chunking import chunk_documents
    from src.vectorstore.faiss_store import FaissStore
    from src.llm.transformers_client import TransformersClient
    from whoosh import index as whoosh_index
    from whoosh.fields import Schema, TEXT, ID

    UPLOAD_DIR = "data/raw"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    docs = load_file(path)
    chunks = chunk_documents(docs)

    # Embeddings + FAISS
    model = TransformersClient()
    texts = [d.page_content for d in chunks]
    embs = model.embed(texts)
    import numpy as np
    if hasattr(embs, "shape"):
        vectors = embs
    else:
        vectors = np.array(embs)

    faiss = FaissStore(dim=384, index_path=cfg['vectorstore']['faiss_index_path'], meta_path=cfg['vectorstore']['faiss_meta_path'])
    metas = [{"source": d.metadata.get("source", ""), "text": d.page_content} for d in chunks]
    faiss.add(vectors, metas)
    faiss.persist()

    # BM25 Whoosh index
    idxdir = "data/index/whoosh"
    os.makedirs(idxdir, exist_ok=True)
    schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True), source=ID(stored=True))
    if not whoosh_index.exists_in(idxdir):
        ix = whoosh_index.create_in(idxdir, schema)
    else:
        ix = whoosh_index.open_dir(idxdir)
    writer = ix.writer()
    for i, d in enumerate(chunks):
        writer.add_document(id=str(i), content=d.page_content, source=d.metadata.get("source", ""))
    writer.commit()

    return {"status": "indexed", "chunks": len(chunks)}

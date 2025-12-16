from fastapi import APIRouter, HTTPException, Request
from rag_backend.schemas.query import Query, ChatRequest
from rag_backend.services.rag_service import RAGService
from fastapi import UploadFile, File

router = APIRouter()


from configs.loader import load_config

cfg = load_config("default.yaml")
rag = RAGService(cfg)

@router.post("/ask")
async def ask(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="Missing question")

    answer = rag.answer(q.question)
    return {"answer": answer}

@router.post("/chat")
async def chat(req: ChatRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="Missing question")

  
    history_text = ""
    for msg in req.history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = (
        "You are an AI assistant that answers strictly from context.\n"
        f"{history_text}\n"
        f"User question: {req.question}"
    )

    answer = rag.answer(prompt)
    return {"answer": answer}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
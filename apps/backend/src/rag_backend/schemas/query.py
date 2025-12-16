from pydantic import BaseModel

class Query(BaseModel):
    question: str
    
class ChatRequest(BaseModel):
    question: str
    history: list

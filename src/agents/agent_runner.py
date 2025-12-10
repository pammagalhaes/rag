from typing import List
from src.llm.base import ModelClient

class AgentRunner:
    def __init__(self, model_client: ModelClient, retriever, max_steps: int = 3):
        self.model = model_client
        self.retriever = retriever
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        k = 3
        for step in range(self.max_steps):
            retrieved = self.retriever.hybrid(question, k=k)
            context = "\n\n".join([r["text"] for r in retrieved])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer based only on the context:"
            answer = self.model.generate(prompt)
            if "Não encontrei a resposta" not in answer:
                return answer
            k *= 2
        return "Desculpe, não encontrei a resposta nos documentos indexados."

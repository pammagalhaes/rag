from typing import List
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from .base import ModelClient


class TransformersClient(ModelClient):
    def __init__(self, embed_model_name="sentence-transformers/all-MiniLM-L6-v2", gen_model_name="google/flan-t5-base", device: int = -1):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.gen = pipeline("text2text-generation", model=gen_model_name, device=device)


    def embed(self, texts: List[str]):
        embs = self.embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs


    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        out = self.gen(prompt, max_length=max_tokens, do_sample=False)
        return out[0]["generated_text"]
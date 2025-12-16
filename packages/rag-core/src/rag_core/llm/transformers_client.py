from typing import List
from openai import OpenAI
from .base import ModelClient
import numpy as np
import os


class TransformersClient(ModelClient):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=api_key)  

    def embed(self, texts: List[str]):
        """
        Uses OpenAI embedding model: text-embedding-3-small
        """
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Uses OpenAI generation model (recommended API).
        """
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=0,
        )
        return response.output_text


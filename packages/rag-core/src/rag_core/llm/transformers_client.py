from typing import List
from openai import OpenAI
from .base import ModelClient
import numpy as np
import os


class TransformersClient(ModelClient):
    """OpenAI client used for embeddings and text generation."""

    DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
    DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"

    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")

        if not openai_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Configure it before instantiating TransformersClient."
            )

        self.backend = "openai"
        self.client = OpenAI(api_key=openai_key)
        self.embed_model = os.getenv(
            "OPENAI_EMBED_MODEL", self.DEFAULT_OPENAI_EMBED_MODEL
        )
        self.chat_model = os.getenv(
            "OPENAI_CHAT_MODEL", self.DEFAULT_OPENAI_CHAT_MODEL
        )

    def embed(self, texts: List[str]):
        """Embed a batch of texts using the configured backend."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.embed_model,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors)

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using the OpenAI Responses API."""
        response = self.client.responses.create(
            model=self.chat_model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=0,
        )
        return response.output_text

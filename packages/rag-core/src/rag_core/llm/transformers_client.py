from typing import List
from openai import OpenAI
from .base import ModelClient
import numpy as np
import os


class TransformersClient(ModelClient):
    """LLM/embedding client that auto-routes to OpenRouter or OpenAI based on env.

    Precedence:
      1. If OPENROUTER_API_KEY is set, route through https://openrouter.ai/api/v1
         using OpenAI-compatible chat/completions + embeddings endpoints.
      2. Otherwise, require OPENAI_API_KEY and use the native OpenAI API
         (incl. the newer `responses.create` endpoint for chat).
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
    DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
    DEFAULT_OPENROUTER_EMBED_MODEL = "openai/text-embedding-3-small"
    DEFAULT_OPENROUTER_CHAT_MODEL = "openai/gpt-4o-mini"

    def __init__(self):
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openrouter_key:
            self.backend = "openrouter"
            self.client = OpenAI(
                api_key=openrouter_key,
                base_url=self.OPENROUTER_BASE_URL,
            )
            self.embed_model = os.getenv(
                "OPENROUTER_EMBED_MODEL", self.DEFAULT_OPENROUTER_EMBED_MODEL
            )
            self.chat_model = os.getenv(
                "OPENROUTER_CHAT_MODEL", self.DEFAULT_OPENROUTER_CHAT_MODEL
            )
        elif openai_key:
            self.backend = "openai"
            self.client = OpenAI(api_key=openai_key)
            self.embed_model = os.getenv(
                "OPENAI_EMBED_MODEL", self.DEFAULT_OPENAI_EMBED_MODEL
            )
            self.chat_model = os.getenv(
                "OPENAI_CHAT_MODEL", self.DEFAULT_OPENAI_CHAT_MODEL
            )
        else:
            raise RuntimeError(
                "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. "
                "Export one of them before instantiating TransformersClient."
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
        """Generate text using the configured backend.

        - OpenAI backend uses the `responses.create` endpoint (matches prior behavior).
        - OpenRouter backend uses the OpenAI-compatible `chat.completions` endpoint,
          which is the only chat-style endpoint exposed by OpenRouter.
        """
        if self.backend == "openai":
            response = self.client.responses.create(
                model=self.chat_model,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=0,
            )
            return response.output_text

        # openrouter (chat.completions)
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content or ""
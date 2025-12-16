from abc import ABC, abstractmethod
from typing import List


class ModelClient(ABC):
    @abstractmethod
    def embed(self, texts: List[str]):
        pass


    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        pass
"""Embedding models. The real one runs locally (ONNX via fastembed), the fake one is for tests."""
import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """bge-small: 384 dimensions, fast on CPU, downloaded once and cached."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 64) -> None:
        from fastembed import TextEmbedding  # imported here so tests do not need the model

        self.model = TextEmbedding(model_name=model_name)
        self.dim = dim
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts, batch_size=self.batch_size)]

    def embed_query(self, text: str) -> list[float]:
        # bge expects an instruction prefix for queries; query_embed adds the right one
        return next(iter(self.model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words vectors. No model, no network: only for tests and demos."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

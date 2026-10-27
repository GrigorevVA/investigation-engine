"""Embedding models. The real one runs locally (ONNX via fastembed), the fake one is for tests."""
import hashlib
import math
import os
import re
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def length_order(texts: list[str]) -> list[int]:
    """Positions of texts from shortest to longest.

    A batch is padded to its longest text, so mixing short and long texts wastes compute.
    Embedding in length order measured 1.8x faster on a laptop CPU (6.8 -> 12.5 docs/s)."""
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


class FastEmbedEmbedder:
    """bge-small: 384 dimensions, fast on CPU, downloaded once and cached."""

    def __init__(self, model_name: str, dim: int, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding  # imported here so tests do not need the model

        self.model = TextEmbedding(model_name=model_name, threads=os.cpu_count())
        self.dim = dim
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        order = length_order(texts)
        vectors = self.model.passage_embed([texts[i] for i in order], batch_size=self.batch_size)
        result: list[list[float]] = [[] for _ in texts]
        for i, vector in zip(order, vectors):
            result[i] = vector.tolist()  # back to the caller's order
        return result

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

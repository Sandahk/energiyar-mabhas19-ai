"""Embedding provider interface (no provider is required or called in the offline baseline).

Implementations may wrap a local multilingual model, OpenAI, Gemini or any other service; each
lives behind this protocol, imports its SDK lazily and is never used unless explicitly
configured. Vectors are compared by cosine similarity; scores of different retrievers are
never added directly (see :mod:`mabhas19.retrieval.hybrid`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    name: str  # provider/model identifier recorded with every index
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

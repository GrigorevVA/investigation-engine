import pytest
from qdrant_client import QdrantClient

from app.config import Settings
from app.models import Chunk, Document
from app.search.embeddings import HashEmbedder
from app.search.vector_store import QdrantStore


class FakeLLM:
    """Returns scripted assistant messages in order and records what it was sent."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, json_mode=False, temperature=0.0) -> dict:
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):  # script a provider error
            raise reply
        return reply

    def chat_json(self, messages, temperature=0.0) -> dict:
        import json

        return json.loads(self.chat(messages, json_mode=True)["content"])


def make_chunk(doc_id: str, text: str, title: str = "T", position: int = 0) -> Chunk:
    return Chunk(f"{doc_id}#{position}", doc_id, title, text, position, {"source": "test", "fmt": "txt"})


def make_doc(doc_id: str, text: str, title: str = "T") -> Document:
    return Document(doc_id, title, text, "test.txt", "txt")


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=64)


@pytest.fixture
def store() -> QdrantStore:
    s = QdrantStore(QdrantClient(":memory:"), "test", dim=64)
    s.ensure_collection()
    return s


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, collection="test", embedding_dim=64, llm_api_key="test")

import json

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.api.main import app, get_engine
from app.engine import Engine
from app.rag.llm import LLMUnavailable
from app.search.embeddings import HashEmbedder
from app.search.vector_store import QdrantStore
from tests.conftest import FakeLLM


class DownLLM(FakeLLM):
    def chat(self, *args, **kwargs):
        raise LLMUnavailable("provider down")


@pytest.fixture
def engine(settings) -> Engine:
    store = QdrantStore(QdrantClient(":memory:"), "api", dim=64)
    return Engine(settings, store, HashEmbedder(64), FakeLLM([]))


@pytest.fixture
def api(engine):
    app.dependency_overrides[get_engine] = lambda: engine
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(api, name: str, text: str, method: str = "post", url: str = "/api/documents"):
    return getattr(api, method)(url, files={"file": (name, text.encode(), "text/plain")})


def test_upload_list_update_delete(api):
    r = upload(api, "Lumora Notes.txt", "Lumora was founded in 2014.")
    assert r.status_code == 201 and r.json()["status"] == "added"
    assert api.get("/api/documents").json()[0]["doc_id"] == "file::lumora-notes"
    r = upload(api, "Lumora Notes.txt", "Lumora was founded in 2016.", "put", "/api/documents/file::lumora-notes")
    assert r.json()["status"] == "updated"
    assert api.delete("/api/documents/file::lumora-notes").json()["status"] == "deleted"
    assert api.delete("/api/documents/file::lumora-notes").status_code == 404


def test_duplicate_upload(api):
    upload(api, "a.txt", "Same text.")
    assert upload(api, "b.txt", "Same text.").json() == {"doc_id": "file::b", "status": "duplicate",
                                                          "chunks": 0, "duplicate_of": "file::a"}


def test_unsupported_format(api):
    assert upload(api, "table.csv", "a,b").status_code == 415


def test_search(api):
    upload(api, "drones.txt", "Cargo drones fly in the mountains.")
    hits = api.post("/api/search", json={"query": "cargo drones", "k": 3, "mode": "vector"}).json()
    assert hits[0]["doc_id"] == "file::drones"


def test_ask(api, engine):
    upload(api, "drones.txt", "Cargo drones fly in the mountains.")
    engine.llm.replies = [
        {"role": "assistant", "content": json.dumps({"subquestions": []})},
        {"role": "assistant", "content": json.dumps({"answer": "In the mountains [C1].", "citations": ["C1"],
                                                     "contradictions": [], "insufficient_evidence": False})},
    ]
    body = api.post("/api/ask", json={"question": "Where do drones fly?", "mode": "vector"}).json()
    assert body["citations"][0]["doc_id"] == "file::drones"


def test_llm_outage_returns_503(api, engine):
    upload(api, "drones.txt", "Cargo drones fly in the mountains.")
    engine.llm = DownLLM([])
    r = api.post("/api/ask", json={"question": "Where do drones fly?"})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_validation(api):
    assert api.post("/api/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_health(api):
    assert api.get("/api/health").json()["status"] == "ok"


def test_ui_page_is_served(api):
    r = api.get("/")
    assert r.status_code == 200 and "AI Investigation Engine" in r.text

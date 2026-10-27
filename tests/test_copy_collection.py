from qdrant_client import QdrantClient

from app.search.vector_store import QdrantStore
from scripts.copy_collection import copy
from tests.conftest import make_chunk


def test_copy_moves_vectors_and_payload(store, embedder):
    chunks = [make_chunk(f"d{i}", f"text number {i}") for i in range(5)]
    store.upsert(chunks, embedder.embed_documents([c.search_text for c in chunks]), ["h"] * 5)
    target = QdrantStore(QdrantClient(":memory:"), "test", dim=64)
    target.ensure_collection()

    assert copy(store.client, target, batch=2) == 5
    assert target.count() == 5
    hit = target.search(embedder.embed_query("text number 3"), k=1)[0]
    assert hit.chunk.doc_id == "d3"

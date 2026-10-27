from app.engine import Engine
from app.search.embeddings import HashEmbedder
from tests.conftest import FakeLLM


def make_engine(settings, store) -> Engine:
    return Engine(settings, store, HashEmbedder(64), FakeLLM([]))


def test_uploaded_file_is_found_by_bm25_with_its_text(settings, store):
    engine = make_engine(settings, store)
    engine.add_file(b"Cargo drones fly over the Caucasus mountains.", "drones.txt")
    engine.wait_for_index()
    hit = engine.search("caucasus", k=1, mode="bm25")[0]
    assert hit.chunk.doc_id == "file::drones"
    assert "Caucasus" in hit.chunk.text  # text fetched from Qdrant, not kept in memory


def test_deleted_file_disappears_even_before_rebuild(settings, store):
    engine = make_engine(settings, store)
    engine.add_file(b"Cargo drones fly over the Caucasus mountains.", "drones.txt")
    engine.wait_for_index()
    engine.pipeline.delete("file::drones")  # delete without scheduling a rebuild: BM25 is stale
    assert engine.search("caucasus", k=3, mode="bm25") == []


def test_restart_rebuilds_state_from_the_store(settings, store):
    make_engine(settings, store).add_file(b"Lumora was founded in 2014.", "lumora.txt")
    restarted = make_engine(settings, store)
    assert [d["doc_id"] for d in restarted.list_documents()] == ["file::lumora"]
    assert restarted.search("lumora founded", k=1, mode="bm25")[0].chunk.doc_id == "file::lumora"

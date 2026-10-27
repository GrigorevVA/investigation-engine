"""Add / unchanged / update / duplicate / delete against an in-memory Qdrant."""
from app.ingest.dedup import content_hash, drop_duplicate_chunks
from app.ingest.pipeline import IngestPipeline
from tests.conftest import make_chunk, make_doc


def pipeline(store, embedder) -> IngestPipeline:
    return IngestPipeline(store, embedder, "paragraph", 800, 100)


def test_hash_ignores_case_and_spacing():
    assert content_hash("Hello   World") == content_hash("hello world\n")
    assert content_hash("hello world") != content_hash("hello there")


def test_drop_duplicate_chunks():
    chunks = [make_chunk("d", "same", position=0), make_chunk("d", "SAME ", position=1), make_chunk("d", "other", position=2)]
    assert [c.position for c in drop_duplicate_chunks(chunks)] == [0, 2]


def test_add_then_unchanged(store, embedder):
    p = pipeline(store, embedder)
    first = p.ingest(make_doc("d1", "Lumora was founded in 2014."))
    assert (first.status, first.chunks) == ("added", 1)
    assert p.ingest(make_doc("d1", "Lumora  was founded in 2014.")).status == "unchanged"
    assert store.count() == 1


def test_update_replaces_old_chunks(store, embedder):
    p = pipeline(store, embedder)
    p.ingest(make_doc("d1", "A" * 500 + "\n\n" + "B" * 500))
    assert store.count() == 2
    result = p.ingest(make_doc("d1", "Only one short paragraph now."))
    assert result.status == "updated"
    assert store.count() == 1  # the second old chunk is gone, not orphaned


def test_same_content_under_new_name_is_duplicate(store, embedder):
    p = pipeline(store, embedder)
    p.ingest(make_doc("d1", "Identical text."))
    result = p.ingest(make_doc("d2", "Identical text."))
    assert (result.status, result.duplicate_of) == ("duplicate", "d1")
    assert store.count() == 1


def test_delete(store, embedder):
    p = pipeline(store, embedder)
    p.ingest(make_doc("d1", "Some text."))
    assert p.delete("d1") is True
    assert store.count() == 0
    assert p.delete("d1") is False


def test_batch_ingest_and_iter_chunks(store, embedder):
    p = pipeline(store, embedder)
    assert p.ingest_batch([make_doc("a", "alpha text"), make_doc("b", "beta text")]) == 2
    stored = {chunk.doc_id: digest for chunk, digest in store.iter_chunks()}
    assert stored == {"a": content_hash("alpha text"), "b": content_hash("beta text")}


def test_vector_search_finds_matching_chunk(store, embedder):
    p = pipeline(store, embedder)
    p.ingest_batch([make_doc("a", "drones carry cargo in mountains"), make_doc("b", "magazines about women")])
    hits = store.search(embedder.embed_query("cargo drones"), k=1)
    assert hits[0].chunk.doc_id == "a"

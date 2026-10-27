import pytest

from app.ingest.pipeline import IngestPipeline
from app.search.hybrid import HybridSearcher, reciprocal_rank_fusion
from tests.conftest import make_doc


def test_rrf_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_order_rewards_agreement():
    ids = [cid for cid, _ in reciprocal_rank_fusion([["x", "y"], ["y", "z"]], k=60)]
    assert ids[0] == "y"  # found by both retrievers


def test_rrf_sorted_descending():
    scores = [s for _, s in reciprocal_rank_fusion([["a", "b", "c"], ["d", "a"]], k=10)]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []


def test_searcher_falls_back_to_vector_without_bm25(store, embedder):
    IngestPipeline(store, embedder, "paragraph", 800, 100).ingest_batch([make_doc("a", "cargo drones")])
    hits = HybridSearcher(store, embedder, bm25=None).search("drones", k=3, mode="hybrid")
    assert hits[0].retriever == "vector"


def test_searcher_rejects_unknown_mode(store, embedder):
    with pytest.raises(ValueError):
        HybridSearcher(store, embedder, bm25=None).search("x", 3, mode="magic")

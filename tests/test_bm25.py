"""Spec for the author's BM25. These tests fail until app/search/bm25.py is implemented."""
import math

import pytest

from app.search.bm25 import BM25Index, tokenize
from tests.conftest import make_chunk

CHUNKS = [
    make_chunk("a", "drone payload drone battery", title="Atlas"),
    make_chunk("b", "magazine for women published monthly", title="First for Women"),
    make_chunk("c", "drone review in the mountains", title="Review"),
    make_chunk("d", "supplier agreement for carbon frames", title="Agreement"),
]


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Drone, and THE battery!") == ["drone", "battery"]


def test_len():
    assert len(BM25Index(CHUNKS)) == 4


def test_best_match_first():
    hits = BM25Index(CHUNKS).search("drone battery", k=4)
    assert hits[0].chunk.doc_id == "a"
    assert hits[0].retriever == "bm25"


def test_scores_are_sorted_and_positive():
    scores = [h.score for h in BM25Index(CHUNKS).search("drone", k=10)]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_chunks_without_query_terms_are_not_returned():
    hits = BM25Index(CHUNKS).search("drone", k=10)
    assert {h.chunk.doc_id for h in hits} == {"a", "c"}


def test_title_is_indexed():
    assert BM25Index(CHUNKS).search("atlas", k=1)[0].chunk.doc_id == "a"


def test_rare_term_scores_higher_than_common_term():
    index = BM25Index(CHUNKS)
    common = index.search("drone", k=10)  # in 2 of 4 chunks
    rare = index.search("carbon", k=10)  # in 1 of 4 chunks
    assert rare[0].score > max(h.score for h in common if h.chunk.doc_id == "c")


def test_k_limits_results():
    assert len(BM25Index(CHUNKS).search("drone magazine supplier", k=2)) == 2


def test_unknown_word_returns_nothing():
    assert BM25Index(CHUNKS).search("zebra", k=5) == []


def test_exact_score_single_term():
    # one chunk, one term, so the score can be checked by hand against the formula
    chunk = make_chunk("x", "alpha beta", title="t")  # search_text "t\nalpha beta" -> ["t","alpha","beta"]
    other = make_chunk("y", "gamma delta", title="t")
    index = BM25Index([chunk, other], k1=1.5, b=0.75)
    n, df, tf, dl, avgdl = 2, 1, 1, 3, 3
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert index.search("alpha", k=1)[0].score == pytest.approx(expected)


def test_empty_index():
    assert BM25Index([]).search("drone", k=3) == []

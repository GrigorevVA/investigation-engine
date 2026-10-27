import pytest

from eval.metrics import (citation_accuracy, contains_reference, exact_match, reciprocal_rank, recall_at_k,
                          token_f1, unique_in_order)


def test_unique_in_order():
    assert unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


# ---- the author's metrics (fail until eval/metrics.py is implemented) ----
def test_recall_at_k():
    assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5
    assert recall_at_k(["a", "x", "b"], ["a", "b"], k=3) == 1.0
    assert recall_at_k(["x"], ["a"], k=5) == 0.0
    assert recall_at_k(["a"], [], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_citation_accuracy():
    assert citation_accuracy(["a", "x"], ["a", "b"]) == 0.5
    assert citation_accuracy(["a", "b"], ["a", "b"]) == 1.0
    assert citation_accuracy([], ["a"]) == 0.0


# ---- answer metrics ----
def test_exact_match_normalizes():
    assert exact_match("The Eiffel Tower!", "eiffel tower") == 1.0


def test_token_f1():
    assert token_f1("Nino Beridze founded it", "Nino Beridze") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert token_f1("nothing", "Nino") == 0.0


def test_contains_reference():
    assert contains_reference("It was founded by Nino Beridze in 2014 [C1].", "Nino Beridze") == 1.0
    assert contains_reference("Founded in 2016.", "2014") == 0.0

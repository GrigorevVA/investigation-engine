"""Evaluation metrics.

TODO(Vladimir): recall_at_k, reciprocal_rank and citation_accuracy are yours to write.
Tests: tests/test_metrics.py. Explanations in Russian: docs/WALKTHROUGH_ru.md.
All ids here are document ids (several chunks of one document count as one document).
"""
import re
import string
from collections import Counter


def unique_in_order(ids: list[str]) -> list[str]:
    """['a', 'b', 'a', 'c'] -> ['a', 'b', 'c']. Use it before ranking metrics."""
    seen: set[str] = set()
    return [x for x in ids if not (x in seen or seen.add(x))]


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Share of relevant documents found in the first k retrieved (after unique_in_order).
    relevant = ['a', 'b'], retrieved top-k contains only 'a'  ->  0.5. Empty relevant -> 0.0."""
    raise NotImplementedError


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / (1-based rank of the FIRST relevant document), 0.0 if none was retrieved.
    MRR is the mean of this value over all questions."""
    raise NotImplementedError


def citation_accuracy(cited: list[str], relevant: list[str]) -> float:
    """Share of cited documents that are relevant (precision of citations).
    No citations -> 0.0. Example: cited ['a', 'x'], relevant ['a', 'b'] -> 0.5."""
    raise NotImplementedError


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, no punctuation, no articles, single spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Word-overlap F1 between answer and reference, as in the HotpotQA evaluation."""
    pred, ref = normalize_answer(prediction).split(), normalize_answer(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_reference(prediction: str, reference: str) -> float:
    """1.0 if the normalized reference appears inside the normalized answer.
    Our answers are full sentences, so this is fairer than exact match for short references."""
    ref = normalize_answer(reference)
    return float(bool(ref) and ref in normalize_answer(prediction))

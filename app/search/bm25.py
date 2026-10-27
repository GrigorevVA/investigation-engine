"""Own keyword search: Okapi BM25.

TODO(Vladimir): implement BM25Index. The tests in tests/test_bm25.py describe the
expected behaviour; make them green. Explanation in Russian: docs/WALKTHROUGH_ru.md.

Formula for one query term t and one chunk d:

    idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    score(t, d) = idf(t) * tf(t,d) * (k1 + 1) / ( tf(t,d) + k1 * (1 - b + b * |d| / avgdl) )

    N      number of chunks in the index
    df(t)  number of chunks that contain t
    tf     how many times t occurs in d
    |d|    length of d in tokens, avgdl = average length over all chunks
    k1     how fast repeated words stop adding score (1.2 to 2.0)
    b      how much long chunks are penalised (0 = never, 1 = fully)

score(query, d) is the sum over the query's tokens. Index Chunk.search_text, not Chunk.text.
"""
import re

from app.models import Chunk, SearchHit

STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his in is it its of on or she that the "
    "their they this to was were which who will with what when where how did does do".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, stopwords removed."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        # Precompute here: tokens of each chunk, document frequency of each term, avgdl.
        raise NotImplementedError("BM25Index is the author's own algorithm, see the module docstring")

    def __len__(self) -> int:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Top k chunks by BM25 score, highest first, only chunks with score > 0.
        Return SearchHit(chunk=..., score=..., retriever="bm25")."""
        raise NotImplementedError

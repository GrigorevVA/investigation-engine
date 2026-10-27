"""Own keyword search: Okapi BM25.

Tests: tests/test_bm25.py. Explanation in Russian: docs/WALKTHROUGH_ru.md.

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
import heapq
import math
import re
from collections import Counter

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
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_lens: list[int] = []
        # inverted index: term -> [(chunk position in self.chunks, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk.search_text)
            self.doc_lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((i, tf))
        self.avgdl = sum(self.doc_lens) / len(chunks) if chunks else 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, []))
        n = len(self.chunks)
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Top k chunks by BM25 score, highest first, only chunks with score > 0."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for i, tf in self.postings.get(term, []):
                norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        best = heapq.nlargest(k, scores.items(), key=lambda pair: pair[1])
        return [SearchHit(chunk=self.chunks[i], score=s, retriever="bm25") for i, s in best if s > 0]

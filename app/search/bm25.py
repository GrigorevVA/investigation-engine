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

Storage (to fit a 512 MB free host): an inverted index in CSR form, three flat numpy arrays instead of
millions of Python tuples. For term id t its postings are positions offsets[t] .. offsets[t+1] in
doc_ids (which chunks contain t) and tfs (how often). 66k chunks: ~250 MB as Python lists, ~40 MB here.
"""
import re
from array import array
from collections import Counter
from collections.abc import Iterable

import numpy as np

from app.models import Chunk, SearchHit

STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his in is it its of on or she that the "
    "their they this to was were which who will with what when where how did does do".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, stopwords removed."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75, drop_text: bool = False) -> None:
        """drop_text=True keeps only (chunk_id, doc_id) of each chunk: hits come back without text,
        title or metadata, and the caller fetches them by chunk_id (saves ~60 MB on 78k chunks)."""
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk | tuple[str, str]] = []
        self.vocab: dict[str, int] = {}  # term -> term id
        term_ids, doc_ids, tfs, lengths = array("i"), array("i"), array("i"), array("i")
        for chunk in chunks:
            tokens = tokenize(chunk.search_text)
            position = len(self.chunks)
            self.chunks.append((chunk.chunk_id, chunk.doc_id) if drop_text else chunk)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                term_ids.append(self.vocab.setdefault(term, len(self.vocab)))
                doc_ids.append(position)
                tfs.append(tf)
        # frombuffer views the arrays without copying them: peak memory matters on a 512 MB host
        self._build_csr(np.frombuffer(term_ids, dtype=np.int32), np.frombuffer(doc_ids, dtype=np.int32),
                        np.frombuffer(tfs, dtype=np.int32), np.array(lengths, dtype=np.float64))

    def _build_csr(self, term_ids: np.ndarray, doc_ids: np.ndarray, tfs: np.ndarray, lengths: np.ndarray) -> None:
        """Group postings by term: sort by term id, then offsets mark where each term starts."""
        order = np.argsort(term_ids, kind="stable")
        self.doc_ids = doc_ids[order]
        self.tfs = np.minimum(tfs[order], 65535).astype(np.uint16)  # no word repeats 65k times in a chunk
        del order
        counts = np.bincount(term_ids, minlength=len(self.vocab))
        self.offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.avgdl = float(lengths.mean()) if len(lengths) else 0.0
        # the length part of the BM25 denominator depends only on the chunk, so compute it once
        self.norm = self.k1 * (1 - self.b + self.b * lengths / self.avgdl) if self.avgdl else lengths

    def __len__(self) -> int:
        return len(self.chunks)

    def idf(self, term: str) -> float:
        t = self.vocab.get(term)
        df = 0 if t is None else int(self.offsets[t + 1] - self.offsets[t])
        n = len(self.chunks)
        return float(np.log((n - df + 0.5) / (df + 0.5) + 1))

    def search(self, query: str, k: int) -> list[SearchHit]:
        """Top k chunks by BM25 score, highest first, only chunks with score > 0."""
        if k <= 0:
            return []
        scores = np.zeros(len(self.chunks))
        for term in set(tokenize(query)):
            t = self.vocab.get(term)
            if t is None:
                continue
            docs = self.doc_ids[self.offsets[t]:self.offsets[t + 1]]
            tf = self.tfs[self.offsets[t]:self.offsets[t + 1]].astype(np.float64)
            scores[docs] += self.idf(term) * tf * (self.k1 + 1) / (tf + self.norm[docs])
        found = np.flatnonzero(scores > 0)
        if len(found) > k:  # partial sort: only the k best need ordering
            found = found[np.argpartition(-scores[found], k - 1)[:k]]
        best = found[np.argsort(-scores[found], kind="stable")]
        return [SearchHit(chunk=self._chunk(i), score=float(scores[i]), retriever="bm25") for i in best]

    def _chunk(self, i: int) -> Chunk:
        stored = self.chunks[i]
        if isinstance(stored, Chunk):
            return stored
        chunk_id, doc_id = stored
        return Chunk(chunk_id=chunk_id, doc_id=doc_id, title="", text="", position=0)

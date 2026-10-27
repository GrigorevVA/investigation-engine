"""Hybrid retrieval: vector search + BM25, merged by Reciprocal Rank Fusion."""
import logging

from app.models import SearchHit
from app.search.bm25 import BM25Index
from app.search.embeddings import Embedder
from app.search.vector_store import QdrantStore

log = logging.getLogger(__name__)
SEARCH_MODES = ("vector", "bm25", "hybrid")


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Own fusion algorithm.

    Each ranking is a list of chunk ids, best first. A chunk at 1-based rank r in one list
    gets 1 / (k + r) from that list; its fused score is the sum over all lists it appears in.
    Return (chunk_id, score) pairs sorted by score, highest first.
    Why ranks and not raw scores: cosine similarity and BM25 live on different scales.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class HybridSearcher:
    def __init__(self, store: QdrantStore, embedder: Embedder, bm25: BM25Index | None,
                 candidate_k: int = 30, rrf_k: int = 60) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int, mode: str = "hybrid") -> list[SearchHit]:
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        if self.bm25 is None and mode != "vector":
            log.warning("BM25 index is not available, falling back to vector search")
            mode = "vector"
        if mode == "vector":
            return self.store.search(self.embedder.embed_query(query), k)
        if mode == "bm25":
            return self._with_text(self.bm25.search(query, k))
        return self._with_text(self._hybrid(query, k))

    def _with_text(self, hits: list[SearchHit]) -> list[SearchHit]:
        """BM25 keeps chunks without text to save memory: fetch the texts of the hits from Qdrant.
        A hit whose chunk was deleted after the BM25 index was built is dropped."""
        missing = [h.chunk.chunk_id for h in hits if not h.chunk.text]
        full = self.store.get_chunks(missing)
        result = []
        for h in hits:
            if h.chunk.text:
                result.append(h)
            elif h.chunk.chunk_id in full:
                result.append(SearchHit(chunk=full[h.chunk.chunk_id], score=h.score, retriever=h.retriever))
        return result

    def _hybrid(self, query: str, k: int) -> list[SearchHit]:
        vector_hits = self.store.search(self.embedder.embed_query(query), self.candidate_k)
        bm25_hits = self.bm25.search(query, self.candidate_k)
        by_id = {h.chunk.chunk_id: h.chunk for h in vector_hits + bm25_hits}
        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in vector_hits], [h.chunk.chunk_id for h in bm25_hits]], self.rrf_k)
        return [SearchHit(chunk=by_id[cid], score=score, retriever="hybrid") for cid, score in fused[:k]]

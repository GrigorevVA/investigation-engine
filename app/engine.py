"""Wires all parts together. The API and the UI talk only to Engine."""
import logging
import tempfile
import threading
from pathlib import Path

from app.config import Settings
from app.ingest.loaders import load_file
from app.ingest.pipeline import IngestPipeline, IngestResult
from app.models import Chunk, SearchHit
from app.rag.answer import InvestigationResult, Investigator
from app.rag.llm import LLMClient
from app.search.bm25 import BM25Index
from app.search.embeddings import Embedder
from app.search.hybrid import HybridSearcher
from app.search.vector_store import QdrantStore

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Settings, store: QdrantStore, embedder: Embedder, llm: LLMClient) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.pipeline = IngestPipeline(store, embedder, settings.chunk_strategy,
                                       settings.chunk_size, settings.chunk_overlap)
        self._chunks: dict[str, Chunk] = {}  # chunk_id -> chunk, the BM25 corpus
        self._bm25: BM25Index | None = None
        self._bm25_dirty = True
        self._lock = threading.Lock()
        store.ensure_collection()
        self._load_chunks()

    def _load_chunks(self) -> None:
        """Rebuild the in-memory view from Qdrant, so a restart loses nothing."""
        self._chunks = {chunk.chunk_id: chunk for chunk, _ in self.store.iter_chunks()}
        self._bm25_dirty = True
        log.info("Loaded %d chunks from the vector store", len(self._chunks))

    def _bm25_index(self) -> BM25Index | None:
        with self._lock:
            if self._bm25_dirty:
                try:
                    self._bm25 = BM25Index(list(self._chunks.values()), self.settings.bm25_k1, self.settings.bm25_b)
                except NotImplementedError:
                    self._bm25 = None
                self._bm25_dirty = False
            return self._bm25

    def searcher(self) -> HybridSearcher:
        return HybridSearcher(self.store, self.embedder, self._bm25_index(),
                              self.settings.candidate_k, self.settings.rrf_k)

    # ---- documents ----
    def add_file(self, data: bytes, filename: str) -> IngestResult:
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            doc = load_file(tmp_path, original_name=filename)
        finally:
            tmp_path.unlink(missing_ok=True)
        result = self.pipeline.ingest(doc)
        if result.status in ("added", "updated"):
            self._forget(doc.doc_id)
            self._chunks.update({c.chunk_id: c for c in result.stored_chunks})
            self._bm25_dirty = True
        return result

    def delete(self, doc_id: str) -> bool:
        deleted = self.pipeline.delete(doc_id)
        if deleted:
            self._forget(doc_id)
            self._bm25_dirty = True
        return deleted

    def _forget(self, doc_id: str) -> None:
        for chunk_id in [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]:
            del self._chunks[chunk_id]

    def list_documents(self, include_corpus: bool = False) -> list[dict]:
        docs: dict[str, dict] = {}
        for chunk in self._chunks.values():
            source = chunk.metadata.get("source", "")
            if not include_corpus and source == "hotpotqa":
                continue
            entry = docs.setdefault(chunk.doc_id, {"doc_id": chunk.doc_id, "title": chunk.title,
                                                   "source": source, "fmt": chunk.metadata.get("fmt", ""),
                                                   "chunks": 0})
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["title"].lower())

    def stats(self) -> dict:
        doc_ids = {c.doc_id for c in self._chunks.values()}
        bm25 = self._bm25_index()
        return {"documents": len(doc_ids), "chunks": len(self._chunks), "collection": self.store.collection,
                "chunk_strategy": self.settings.chunk_strategy, "search_mode": self.settings.search_mode,
                "bm25_ready": bm25 is not None, "llm_model": self.settings.llm_model}

    # ---- search and investigation ----
    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[SearchHit]:
        return self.searcher().search(query, k or self.settings.top_k, mode or self.settings.search_mode)

    def investigate(self, question: str, mode: str | None = None) -> InvestigationResult:
        s = self.settings
        investigator = Investigator(self.llm, self.searcher(), s.top_k, mode or s.search_mode,
                                    s.max_subquestions, s.max_context_chunks, s.max_tool_rounds)
        return investigator.investigate(question)


def build_engine(settings: Settings) -> Engine:
    from app.search.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    return Engine(settings, QdrantStore.from_settings(settings), embedder, LLMClient.from_settings(settings))

"""Wires all parts together. The API and the UI talk only to Engine."""
import ctypes
import logging
import tempfile
import threading
import time
from collections.abc import Iterator
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
CORPUS_SOURCE = "hotpotqa"


class Engine:
    """Holds no chunk texts in memory (512 MB free hosting): a document registry for the UI,
    a BM25 index over text-less chunks, and Qdrant for everything else."""

    def __init__(self, settings: Settings, store: QdrantStore, embedder: Embedder, llm: LLMClient) -> None:
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.pipeline = IngestPipeline(store, embedder, settings.chunk_strategy,
                                       settings.chunk_size, settings.chunk_overlap)
        self._docs: dict[str, dict] = {}  # uploaded and synthetic docs: doc_id -> title, source, fmt, chunks
        self._corpus = {"documents": 0, "chunks": 0}  # 66k corpus paragraphs are counted, not listed
        self._bm25: BM25Index | None = None
        self._lock = threading.Lock()
        self._rebuilding = False
        self._rebuild_pending = False
        store.ensure_collection()
        self._load_index()

    def _load_index(self) -> None:
        """Read every chunk from Qdrant once: build BM25 and the document registry, then swap them in.
        Also how a restart loses nothing: the state is rebuilt from the database."""
        docs: dict[str, dict] = {}
        corpus_docs: set[str] = set()
        corpus_chunks = 0

        def stream() -> Iterator[Chunk]:
            nonlocal corpus_chunks
            for chunk, _ in self.store.iter_chunks():
                if chunk.metadata.get("source") == CORPUS_SOURCE:
                    corpus_docs.add(chunk.doc_id)
                    corpus_chunks += 1
                else:
                    _register(docs, chunk)
                yield chunk

        bm25 = BM25Index(stream(), self.settings.bm25_k1, self.settings.bm25_b, drop_text=True)
        with self._lock:
            self._bm25, self._docs = bm25, docs
            self._corpus = {"documents": len(corpus_docs), "chunks": corpus_chunks}
        _release_free_memory()
        log.info("Indexed %d chunks for BM25 (%d corpus documents, %d others)", len(bm25), len(corpus_docs), len(docs))

    def _schedule_rebuild(self) -> None:
        """Rebuild BM25 in a background thread after a change (~40 s for 78k chunks from Qdrant Cloud).
        Meanwhile hybrid search falls back to vector search, which sees the change at once.
        Changes made during a rebuild queue one more rebuild."""
        with self._lock:
            self._rebuild_pending = True
            if self._rebuilding:
                return
            self._rebuilding = True
        threading.Thread(target=self._rebuild_loop, daemon=True).start()

    def _rebuild_loop(self) -> None:
        while True:
            with self._lock:
                if not self._rebuild_pending:
                    self._rebuilding = False
                    return
                self._rebuild_pending = False
            try:
                with self._lock:
                    self._bm25 = None  # free the old index first: two at once do not fit in 512 MB
                self._load_index()
            except Exception:  # a failed rebuild must not kill the thread state; the old index stays
                log.exception("BM25 rebuild failed, keeping the previous index")

    def wait_for_index(self, timeout_s: float = 30.0) -> None:
        """Block until a scheduled rebuild has finished (tests, scripts)."""
        deadline = time.monotonic() + timeout_s
        while self._rebuilding and time.monotonic() < deadline:
            time.sleep(0.01)

    def searcher(self) -> HybridSearcher:
        return HybridSearcher(self.store, self.embedder, self._bm25, self.settings.candidate_k, self.settings.rrf_k)

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
            with self._lock:
                self._docs.pop(doc.doc_id, None)
                for chunk in result.stored_chunks:
                    _register(self._docs, chunk)
            self._schedule_rebuild()
        return result

    def delete(self, doc_id: str) -> bool:
        deleted = self.pipeline.delete(doc_id)
        if deleted:
            with self._lock:
                self._docs.pop(doc_id, None)
            self._schedule_rebuild()
        return deleted

    def list_documents(self) -> list[dict]:
        """Uploaded and synthetic documents. Corpus paragraphs are only counted in stats()."""
        return sorted(self._docs.values(), key=lambda d: d["title"].lower())

    def stats(self) -> dict:
        own_chunks = sum(d["chunks"] for d in self._docs.values())
        return {"documents": len(self._docs) + self._corpus["documents"],
                "chunks": own_chunks + self._corpus["chunks"], "corpus_documents": self._corpus["documents"],
                "collection": self.store.collection, "chunk_strategy": self.settings.chunk_strategy,
                "search_mode": self.settings.search_mode, "bm25_ready": self._bm25 is not None,
                "bm25_rebuilding": self._rebuilding, "llm_model": self.settings.llm_model}

    # ---- search and investigation ----
    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[SearchHit]:
        return self.searcher().search(query, k or self.settings.top_k, mode or self.settings.search_mode)

    def investigate(self, question: str, mode: str | None = None) -> InvestigationResult:
        s = self.settings
        investigator = Investigator(self.llm, self.searcher(), s.top_k, mode or s.search_mode,
                                    s.max_subquestions, s.max_context_chunks, s.max_tool_rounds)
        return investigator.investigate(question)


def _release_free_memory() -> None:
    """Ask glibc to hand freed heap pages back to the OS after a big build (Linux only, no-op elsewhere)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _register(docs: dict[str, dict], chunk: Chunk) -> None:
    entry = docs.setdefault(chunk.doc_id, {"doc_id": chunk.doc_id, "title": chunk.title,
                                           "source": chunk.metadata.get("source", ""),
                                           "fmt": chunk.metadata.get("fmt", ""), "chunks": 0})
    entry["chunks"] += 1


def build_engine(settings: Settings) -> Engine:
    from app.search.embeddings import FastEmbedEmbedder

    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    return Engine(settings, QdrantStore.from_settings(settings), embedder, LLMClient.from_settings(settings))

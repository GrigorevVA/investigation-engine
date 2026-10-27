"""Add, update and delete documents: chunk, deduplicate, embed, store."""
from dataclasses import dataclass, field

from app.ingest.chunking import chunk_document
from app.ingest.dedup import content_hash, drop_duplicate_chunks
from app.models import Chunk, Document
from app.search.embeddings import Embedder
from app.search.vector_store import QdrantStore


@dataclass
class IngestResult:
    doc_id: str
    status: str  # "added" | "updated" | "unchanged" | "duplicate"
    chunks: int = 0
    duplicate_of: str | None = None
    stored_chunks: list[Chunk] = field(default_factory=list, repr=False)


class IngestPipeline:
    def __init__(self, store: QdrantStore, embedder: Embedder, strategy: str, size: int, overlap: int) -> None:
        self.store = store
        self.embedder = embedder
        self.strategy = strategy
        self.size = size
        self.overlap = overlap

    def ingest(self, doc: Document) -> IngestResult:
        """Decision table:
        same doc_id, same content       -> unchanged, nothing to do
        same doc_id, new content        -> updated, old chunks replaced
        other doc_id, same content      -> duplicate, skipped
        new doc_id, new content         -> added
        """
        digest = content_hash(doc.text)
        stored_hash = self.store.doc_hash(doc.doc_id)
        if stored_hash == digest:
            return IngestResult(doc.doc_id, "unchanged")
        if stored_hash is None:
            twin = self.store.find_doc_by_hash(digest)
            if twin is not None:
                return IngestResult(doc.doc_id, "duplicate", duplicate_of=twin)
        else:
            self.store.delete_doc(doc.doc_id)
        chunks = self._store(doc, digest)
        status = "added" if stored_hash is None else "updated"
        return IngestResult(doc.doc_id, status, len(chunks), stored_chunks=chunks)

    def ingest_batch(self, docs: list[Document]) -> int:
        """Bulk path for the dataset: no per-document checks, one embedding call per batch."""
        all_chunks: list[Chunk] = []
        hashes: list[str] = []
        for doc in docs:
            chunks = drop_duplicate_chunks(chunk_document(doc, self.strategy, self.size, self.overlap))
            all_chunks.extend(chunks)
            hashes.extend([content_hash(doc.text)] * len(chunks))
        vectors = self.embedder.embed_documents([c.search_text for c in all_chunks])
        self.store.upsert(all_chunks, vectors, hashes)
        return len(all_chunks)

    def delete(self, doc_id: str) -> bool:
        if self.store.doc_hash(doc_id) is None:
            return False
        self.store.delete_doc(doc_id)
        return True

    def _store(self, doc: Document, digest: str) -> list[Chunk]:
        chunks = drop_duplicate_chunks(chunk_document(doc, self.strategy, self.size, self.overlap))
        vectors = self.embedder.embed_documents([c.search_text for c in chunks])
        self.store.upsert(chunks, vectors, [digest] * len(chunks))
        return chunks

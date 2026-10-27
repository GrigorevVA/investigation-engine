"""Qdrant wrapper. Every Qdrant call goes through here, so errors are handled in one place."""
import uuid
from collections.abc import Callable, Iterator
from typing import TypeVar

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.config import Settings
from app.models import Chunk, SearchHit
from app.retry import RetryableError, with_retry

T = TypeVar("T")


class VectorStoreError(RuntimeError):
    """The vector database is unreachable or rejected the request."""


def point_id(chunk_id: str) -> str:
    """Qdrant needs UUID or int ids; uuid5 gives the same UUID for the same chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str, dim: int, max_retries: int = 3) -> None:
        self.client = client
        self.collection = collection
        self.dim = dim
        self.max_retries = max_retries

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30)
        else:
            client = QdrantClient(path=settings.qdrant_path)  # local mode, files on disk
        return cls(client, settings.collection, settings.embedding_dim)

    def _call(self, fn: Callable[[], T]) -> T:
        def attempt() -> T:
            try:
                return fn()
            except (ResponseHandlingException, httpx.TransportError) as err:
                raise RetryableError(str(err)) from err
            except UnexpectedResponse as err:
                if err.status_code in (429, 500, 502, 503, 504):
                    raise RetryableError(str(err)) from err
                raise VectorStoreError(f"Qdrant rejected the request: {err}") from err

        try:
            return with_retry(attempt, self.max_retries, base_s=0.5)
        except RetryableError as err:
            raise VectorStoreError(f"Qdrant is unavailable: {err}") from err

    def exists(self) -> bool:
        return self._call(lambda: self.client.collection_exists(self.collection))

    def ensure_collection(self) -> None:
        if self.exists():
            return
        self._call(lambda: self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        ))
        for field in ("doc_id", "content_hash"):
            self._call(lambda f=field: self.client.create_payload_index(
                self.collection, f, models.PayloadSchemaType.KEYWORD))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], doc_hashes: list[str]) -> None:
        """doc_hashes[i] is the content hash of the document chunks[i] belongs to."""
        points = [
            models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=_payload(c, h))
            for c, v, h in zip(chunks, vectors, doc_hashes, strict=True)
        ]
        self.upsert_points(points)

    def upsert_points(self, points: list[models.PointStruct]) -> None:
        """Write ready points (vector + payload), 256 per request. Also used to copy collections."""
        for start in range(0, len(points), 256):
            batch = points[start:start + 256]
            self._call(lambda b=batch: self.client.upsert(self.collection, b))

    def search(self, vector: list[float], k: int) -> list[SearchHit]:
        result = self._call(lambda: self.client.query_points(
            self.collection, query=vector, limit=k, with_payload=True))
        return [SearchHit(chunk=_chunk(p.payload), score=p.score, retriever="vector") for p in result.points]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        """Full chunks (with text) by chunk id. Ids that are no longer stored are missing from the result."""
        if not chunk_ids:
            return {}
        points = self._call(lambda: self.client.retrieve(
            self.collection, ids=[point_id(cid) for cid in chunk_ids], with_payload=True))
        return {p.payload["chunk_id"]: _chunk(p.payload) for p in points}

    def delete_doc(self, doc_id: str) -> None:
        self._call(lambda: self.client.delete(self.collection, points_selector=_doc_filter("doc_id", doc_id)))

    def doc_hash(self, doc_id: str) -> str | None:
        """Content hash of a stored document, or None if the document is not stored."""
        payload = self._first(_doc_filter("doc_id", doc_id))
        return payload["content_hash"] if payload else None

    def find_doc_by_hash(self, digest: str) -> str | None:
        payload = self._first(_doc_filter("content_hash", digest))
        return payload["doc_id"] if payload else None

    def _first(self, flt: models.Filter) -> dict | None:
        points, _ = self._call(lambda: self.client.scroll(
            self.collection, scroll_filter=flt, limit=1, with_payload=True))
        return points[0].payload if points else None

    def iter_chunks(self, batch: int = 1000) -> Iterator[tuple[Chunk, str]]:
        """Every stored chunk with its document hash. Used to rebuild BM25 at startup."""
        offset = None
        while True:
            points, offset = self._call(lambda o=offset: self.client.scroll(
                self.collection, limit=batch, offset=o, with_payload=True))
            for p in points:
                yield _chunk(p.payload), p.payload["content_hash"]
            if offset is None:
                return

    def count(self) -> int:
        return self._call(lambda: self.client.count(self.collection).count)


def _doc_filter(key: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])


def _payload(chunk: Chunk, doc_hash: str) -> dict:
    return {
        "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "title": chunk.title,
        "text": chunk.text, "position": chunk.position, "metadata": chunk.metadata,
        "content_hash": doc_hash,
    }


def _chunk(payload: dict) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"], doc_id=payload["doc_id"], title=payload["title"],
        text=payload["text"], position=payload["position"], metadata=payload.get("metadata", {}),
    )

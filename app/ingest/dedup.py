"""Duplicate detection by content hash."""
import hashlib
import re

from app.models import Chunk


def content_hash(text: str) -> str:
    """Hash of the normalized text: case and whitespace differences do not matter."""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def drop_duplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Remove repeated chunks inside one document (copied boilerplate, repeated headers)."""
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        digest = content_hash(chunk.text)
        if digest not in seen:
            seen.add(digest)
            unique.append(chunk)
    return unique

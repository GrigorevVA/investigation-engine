"""Two chunking strategies, compared in the evaluation report.

Strategy A, "fixed": windows of chunk_size characters with overlap, cut at a space.
Strategy B, "paragraph": keep paragraphs whole, merge short ones, split only oversized ones.
"""
from app.models import Chunk, Document


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)  # do not cut a word in half
            end = space if space != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = _next_word_start(text, max(end - overlap, start + 1), limit=end)
    return [c for c in chunks if c]


def _next_word_start(text: str, pos: int, limit: int) -> int:
    """Move pos forward to the start of a word, so the overlap does not begin mid-word."""
    if pos == 0 or text[pos - 1].isspace():
        return pos
    space = text.find(" ", pos, limit)
    return space + 1 if space != -1 else pos


def chunk_paragraph(text: str, max_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_size:  # a huge paragraph falls back to strategy A
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(chunk_fixed(paragraph, max_size, overlap))
        elif len(current) + len(paragraph) + 2 <= max_size:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, strategy: str, size: int, overlap: int) -> list[Chunk]:
    if strategy == "fixed":
        pieces = chunk_fixed(doc.text, size, overlap)
    elif strategy == "paragraph":
        pieces = chunk_paragraph(doc.text, size, overlap)
    else:
        raise ValueError(f"unknown chunk strategy: {strategy}")
    meta = {"source": doc.source, "fmt": doc.fmt, "strategy": strategy}
    return [
        Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, title=doc.title, text=piece, position=i, metadata=meta)
        for i, piece in enumerate(pieces)
    ]

import pytest

from app.ingest.chunking import chunk_document, chunk_fixed, chunk_paragraph
from tests.conftest import make_doc

TEXT = " ".join(f"word{i}" for i in range(200))  # about 1,400 characters


def test_fixed_respects_size_and_overlaps():
    chunks = chunk_fixed(TEXT, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    first_tail = chunks[0].split()[-1]
    assert first_tail in chunks[1]  # overlap keeps the boundary word in both chunks


def test_fixed_does_not_cut_words():
    words = set(TEXT.split())
    for chunk in chunk_fixed(TEXT, size=300, overlap=50):
        assert set(chunk.split()) <= words


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_fixed(TEXT, size=100, overlap=100)


def test_paragraph_merges_short_paragraphs():
    text = "Short one.\n\nShort two.\n\nShort three."
    assert chunk_paragraph(text, max_size=800, overlap=50) == ["Short one.\n\nShort two.\n\nShort three."]


def test_paragraph_keeps_paragraph_boundaries():
    a, b = "A" * 500, "B" * 500
    assert chunk_paragraph(f"{a}\n\n{b}", max_size=800, overlap=50) == [a, b]


def test_paragraph_splits_oversized_paragraph():
    chunks = chunk_paragraph(TEXT, max_size=300, overlap=50)
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)


def test_chunk_document_ids_and_metadata():
    chunks = chunk_document(make_doc("d1", TEXT), "fixed", 300, 50)
    assert [c.chunk_id for c in chunks[:2]] == ["d1#0", "d1#1"]
    assert chunks[0].metadata["strategy"] == "fixed"
    assert chunks[0].search_text.startswith("T\n")


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_document(make_doc("d1", TEXT), "semantic", 300, 50)

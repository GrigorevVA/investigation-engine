import numpy as np

from app.search.embeddings import FastEmbedEmbedder, length_order


class FakeModel:
    """Returns [len(text)] for each text, so the output shows which text it came from."""

    def passage_embed(self, texts: list[str], batch_size: int):
        return (np.array([float(len(t))]) for t in texts)


def test_length_order_shortest_first():
    assert length_order(["ccc", "a", "bb"]) == [1, 2, 0]


def test_embed_documents_keeps_caller_order():
    embedder = FastEmbedEmbedder.__new__(FastEmbedEmbedder)  # skip __init__: no model download
    embedder.model, embedder.batch_size, embedder.dim = FakeModel(), 2, 1
    assert embedder.embed_documents(["ccc", "a", "bb"]) == [[3.0], [1.0], [2.0]]

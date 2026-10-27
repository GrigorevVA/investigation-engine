"""Plain data objects that move between the modules."""
from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str  # file name or dataset name, e.g. "hotpotqa"
    fmt: str  # "txt" | "md" | "pdf" | "docx" | "jsonl"
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}#{position}"
    doc_id: str
    title: str
    text: str
    position: int
    metadata: dict = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        """Text used for embeddings and BM25. The title adds context to short chunks."""
        return f"{self.title}\n{self.text}"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    retriever: str = ""  # "vector" | "bm25" | "hybrid"

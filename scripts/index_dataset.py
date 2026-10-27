"""Index the corpus (and the synthetic documents) into one Qdrant collection.

Run once per configuration on your own machine; the hosted app only reads.
    python -m scripts.index_dataset --strategy paragraph --collection documents_paragraph
    python -m scripts.index_dataset --strategy fixed --size 300 --overlap 50 --collection documents_fixed
Re-running is safe: chunk ids are stable, so points are overwritten, not duplicated.
"""
import argparse
import json
import time
from pathlib import Path

from app.config import get_settings
from app.ingest.loaders import SUPPORTED_FORMATS, load_file
from app.ingest.pipeline import IngestPipeline
from app.models import Document
from app.search.embeddings import FastEmbedEmbedder
from app.search.vector_store import QdrantStore


def read_corpus(path: Path, limit: int | None) -> list[Document]:
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(row["doc_id"], row["title"], row["text"], row["source"], "jsonl"))
            if limit and len(docs) >= limit:
                break
    return docs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.jsonl")
    p.add_argument("--synthetic-dir", default="data/synthetic")
    p.add_argument("--strategy", choices=["paragraph", "fixed"], default="paragraph")
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=120)
    p.add_argument("--collection", default=None)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--start", type=int, default=0, help="skip the first N corpus docs (resume)")
    p.add_argument("--limit", type=int, default=None, help="index only N corpus docs (quick test)")
    args = p.parse_args()

    settings = get_settings().model_copy(update={"collection": args.collection or get_settings().collection})
    store = QdrantStore.from_settings(settings)
    store.ensure_collection()
    embedder = FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
    pipeline = IngestPipeline(store, embedder, args.strategy, args.size, args.overlap)

    for path in sorted(Path(args.synthetic_dir).glob("*")):
        if path.suffix.lower() in SUPPORTED_FORMATS:
            result = pipeline.ingest(load_file(path))
            print(f"{path.name}: {result.status} ({result.chunks} chunks)")

    docs = read_corpus(Path(args.corpus), args.limit)[args.start:]
    started, total_chunks = time.time(), 0
    for i in range(0, len(docs), args.batch):
        total_chunks += pipeline.ingest_batch(docs[i:i + args.batch])
        done = i + min(args.batch, len(docs) - i)
        rate = done / (time.time() - started)
        print(f"{args.start + done:,}/{args.start + len(docs):,} docs, {total_chunks:,} chunks, "
              f"{rate:.0f} docs/s, ~{(len(docs) - done) / rate / 60:.1f} min left", flush=True)
    print(f"Done. Collection {settings.collection!r} now holds {store.count():,} chunks.")


if __name__ == "__main__":
    main()

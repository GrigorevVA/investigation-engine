"""Copy a collection from local Qdrant (data/qdrant) to Qdrant Cloud without re-embedding.

Embedding the corpus takes ~1.5 hours on a laptop; copying ready vectors takes minutes.
QDRANT_URL and QDRANT_API_KEY in .env point to the target cluster.
    python -m scripts.copy_collection --collection documents_paragraph
"""
import argparse

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.search.vector_store import QdrantStore


def copy(source: QdrantClient, target: QdrantStore, batch: int = 512) -> int:
    copied, offset = 0, None
    while True:
        points, offset = source.scroll(target.collection, limit=batch, offset=offset,
                                       with_payload=True, with_vectors=True)
        target.upsert_points([models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        copied += len(points)
        print(f"{copied:,} points copied", flush=True)
        if offset is None:
            return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    args = parser.parse_args()
    settings = get_settings().model_copy(update={"collection": args.collection})
    if not settings.qdrant_url:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY in .env: the target must be Qdrant Cloud")
    target = QdrantStore.from_settings(settings)
    target.ensure_collection()
    total = copy(QdrantClient(path=settings.qdrant_path), target)
    print(f"Done: {total:,} points, target now holds {target.count():,}")


if __name__ == "__main__":
    main()

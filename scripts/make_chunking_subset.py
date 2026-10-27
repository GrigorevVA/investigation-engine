"""Reproducible corpus subset for the chunking comparison (strategy A vs B).

Embedding the full corpus with 300-character chunks takes about 4 hours on a laptop CPU,
so both strategies are compared on the same smaller corpus instead:
the first N paragraphs of data/corpus.jsonl (distractors) + every document an eval question needs.
Same input files -> same subset, no randomness.

    python -m scripts.make_chunking_subset --size 10000
Outputs data/corpus_subset.jsonl.
"""
import argparse
import json
from pathlib import Path


def build_subset(corpus_path: Path, questions_path: Path, size: int) -> list[dict]:
    needed: set[str] = set()
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        needed.update(json.loads(line)["relevant_doc_ids"])
    subset = []
    with corpus_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if i < size or row["doc_id"] in needed:
                subset.append(row)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--out", default="data/corpus_subset.jsonl")
    args = parser.parse_args()
    subset = build_subset(Path(args.corpus), Path(args.questions), args.size)
    with Path(args.out).open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in subset)
    print(f"Subset: {len(subset):,} documents -> {args.out}")


if __name__ == "__main__":
    main()

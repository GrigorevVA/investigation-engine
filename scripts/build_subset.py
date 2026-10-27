"""Download HotpotQA (dev, distractor setting) and turn it into a corpus + questions.

Each question in this file comes with 10 Wikipedia paragraphs (2 gold + 8 distractors).
Unique paragraphs across all 7,405 questions form the corpus.

    python -m scripts.build_subset
Outputs data/corpus.jsonl and data/hotpot_questions.jsonl.
"""
import argparse
import json
import urllib.request
from pathlib import Path

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
MIN_DOCS = 50_000


def download(url: str, target: Path) -> Path:
    if target.exists():
        print(f"Using cached {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} (about 45 MB)...")
    urllib.request.urlretrieve(url, target)
    return target


def build(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    corpus: dict[str, dict] = {}
    questions = []
    for item in raw:
        for title, sentences in item["context"]:
            text = "".join(sentences).strip()
            if text and title not in corpus:  # the same article appears under many questions
                corpus[title] = {"doc_id": f"hotpot::{title}", "title": title, "text": text, "source": "hotpotqa"}
        supporting = sorted({title for title, _ in item["supporting_facts"]})
        questions.append({
            "id": item["_id"], "question": item["question"], "answer": item["answer"],
            "type": item["type"], "level": item["level"],
            "relevant_doc_ids": [f"hotpot::{t}" for t in supporting],
        })
    return list(corpus.values()), questions


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    data = Path(args.data_dir)
    raw_path = download(URL, data / "raw" / "hotpot_dev_distractor_v1.json")
    corpus, questions = build(json.loads(raw_path.read_text(encoding="utf-8")))
    write_jsonl(corpus, data / "corpus.jsonl")
    write_jsonl(questions, data / "hotpot_questions.jsonl")
    print(f"Corpus: {len(corpus):,} unique paragraphs, questions: {len(questions):,}")
    if len(corpus) < MIN_DOCS:
        raise SystemExit(f"Corpus has fewer than {MIN_DOCS:,} documents; the task requires at least that many")


if __name__ == "__main__":
    main()

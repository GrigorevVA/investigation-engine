import json

from scripts.make_chunking_subset import build_subset


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_subset_keeps_head_and_every_needed_doc(tmp_path):
    corpus, questions = tmp_path / "corpus.jsonl", tmp_path / "questions.jsonl"
    write_jsonl(corpus, [{"doc_id": f"d{i}"} for i in range(10)])
    write_jsonl(questions, [{"relevant_doc_ids": ["d7", "file::x"]}, {"relevant_doc_ids": []}])
    ids = [r["doc_id"] for r in build_subset(corpus, questions, size=3)]
    assert ids == ["d0", "d1", "d2", "d7"]

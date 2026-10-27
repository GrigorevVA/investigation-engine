"""Run the evaluation set against one configuration and compare configurations.

    python -m eval.run_eval build                          # once: write eval/questions.jsonl
    python -m eval.run_eval run --name hybrid_paragraph --collection documents_paragraph --mode hybrid
    python -m eval.run_eval run --name vector_paragraph --collection documents_paragraph --mode vector --retrieval-only
    python -m eval.run_eval compare hybrid_paragraph vector_paragraph
Results go to eval/results/<name>.jsonl (per question) and <name>.summary.json.
"""
import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean

from app.config import get_settings
from app.engine import build_engine
from app.rag.llm import LLMUnavailable
from app.rag.prompts import JUDGE_SYSTEM, JUDGE_USER
from app.search.vector_store import QdrantStore
from eval.metrics import (citation_accuracy, contains_reference, reciprocal_rank, recall_at_k, token_f1,
                          unique_in_order)

EVAL_DIR = Path("eval")
RESULTS = EVAL_DIR / "results"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_questions(seed: int = 42, bridge: int = 15, comparison: int = 10) -> None:
    hotpot = read_jsonl(Path("data/hotpot_questions.jsonl"))
    rng = random.Random(seed)
    picked = []
    for kind, n in (("bridge", bridge), ("comparison", comparison)):
        pool = [q for q in hotpot if q["type"] == kind]
        for q in rng.sample(pool, n):
            picked.append({"id": q["id"], "type": "multi_hop" if kind == "bridge" else "comparison",
                           "question": q["question"], "answer": q["answer"],
                           "relevant_doc_ids": q["relevant_doc_ids"]})
    rows = picked + read_jsonl(EVAL_DIR / "custom_questions.jsonl")
    with (EVAL_DIR / "questions.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"Wrote {len(rows)} questions to eval/questions.jsonl")


def retrieval_scores(doc_ids: list[str], relevant: list[str]) -> dict:
    if not relevant:
        return {}
    ranked = unique_in_order(doc_ids)
    return {"recall@5": recall_at_k(ranked, relevant, 5), "recall@10": recall_at_k(ranked, relevant, 10),
            "rr": reciprocal_rank(ranked, relevant)}


def judge(engine, q: dict, answer: str) -> float | None:
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER.format(question=q["question"], reference=q["answer"], answer=answer)}]
    try:
        return float(engine.llm.chat_json(messages).get("score", 0))
    except (LLMUnavailable, ValueError, TypeError):
        return None


def evaluate_one(engine, q: dict, mode: str, retrieval_only: bool, use_judge: bool) -> dict:
    row = {"id": q["id"], "type": q["type"], "question": q["question"], "reference": q["answer"]}
    if retrieval_only:
        hits = engine.search(q["question"], 10, mode)
        row["retrieved"] = [h.chunk.doc_id for h in hits]
        return row | retrieval_scores(row["retrieved"], q["relevant_doc_ids"])
    result = engine.investigate(q["question"], mode)
    cited = unique_in_order([c.doc_id for c in result.citations])
    row |= {"answer": result.answer, "insufficient": result.insufficient_evidence, "cited": cited,
            "retrieved": [c.doc_id for c in result.retrieved], "contradictions": result.contradictions,
            "queries": result.queries}
    row |= retrieval_scores(row["retrieved"], q["relevant_doc_ids"])
    answerable = q["answer"] != "NO_ANSWER"
    row["abstention_correct"] = float(result.insufficient_evidence != answerable)
    if answerable:
        row["contains_ref"] = contains_reference(result.answer, q["answer"])
        row["f1"] = token_f1(result.answer, q["answer"])
        row["citation_acc"] = citation_accuracy(cited, q["relevant_doc_ids"])
    if q.get("expect_contradiction"):
        row["contradiction_found"] = float(bool(result.contradictions))
    if use_judge:
        row["judge"] = judge(engine, q, result.answer)
    return row


def summarize(rows: list[dict]) -> dict:
    keys = ["recall@5", "recall@10", "rr", "contains_ref", "f1", "judge", "citation_acc",
            "abstention_correct", "contradiction_found"]
    summary = {"questions": len(rows), "errors": sum(1 for r in rows if "error" in r)}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            summary["MRR" if key == "rr" else key] = round(mean(values), 3)
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    summary["by_type"] = {t: {k: round(mean(v[k] for v in rs if v.get(k) is not None), 3)
                              for k in ("recall@10", "contains_ref", "abstention_correct")
                              if any(v.get(k) is not None for v in rs)} | {"n": len(rs)}
                          for t, rs in by_type.items()}
    return summary


def run(args: argparse.Namespace) -> None:
    settings = get_settings().model_copy(update={"collection": args.collection, "search_mode": args.mode})
    print(f"Qdrant: {settings.qdrant_url or settings.qdrant_path}, collection {args.collection}", flush=True)
    store = QdrantStore.from_settings(settings)
    if not store.exists():  # check before build_engine, which would create an empty collection
        raise SystemExit(f"Collection {args.collection!r} does not exist here: wrong QDRANT_URL / QDRANT_PATH?")
    store.client.close()  # local Qdrant allows one client per folder
    engine = build_engine(settings)
    questions = read_jsonl(EVAL_DIR / "questions.jsonl")[: args.limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, q in enumerate(questions, 1):
        try:
            rows.append(evaluate_one(engine, q, args.mode, args.retrieval_only, args.judge))
        except LLMUnavailable as err:
            rows.append({"id": q["id"], "type": q["type"], "error": str(err)})
        print(f"[{i}/{len(questions)}] {q['id']} {q['type']}", flush=True)
        if not args.retrieval_only:
            time.sleep(args.sleep)  # stay under free-tier rate limits
    (RESULTS / f"{args.name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                                encoding="utf-8")
    summary = summarize(rows) | {"config": {"collection": args.collection, "mode": args.mode,
                                            "retrieval_only": args.retrieval_only, "model": settings.llm_model}}
    (RESULTS / f"{args.name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def compare(names: list[str]) -> None:
    summaries = {n: json.loads((RESULTS / f"{n}.summary.json").read_text()) for n in names}
    metrics = [k for k in ("recall@5", "recall@10", "MRR", "contains_ref", "f1", "judge", "citation_acc",
                           "abstention_correct", "contradiction_found") if any(k in s for s in summaries.values())]
    print("| metric | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for m in metrics:
        print(f"| {m} | " + " | ".join(str(summaries[n].get(m, "–")) for n in names) + " |")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--collection", default=get_settings().collection)
    r.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    r.add_argument("--retrieval-only", action="store_true")
    r.add_argument("--judge", action="store_true", help="also grade answers with the LLM")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--sleep", type=float, default=2.0)
    c = sub.add_parser("compare")
    c.add_argument("names", nargs="+")
    args = p.parse_args()
    if args.cmd == "build":
        build_questions()
    elif args.cmd == "run":
        run(args)
    else:
        compare(args.names)


if __name__ == "__main__":
    main()

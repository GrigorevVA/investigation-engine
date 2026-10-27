# Evaluation Report

_Fill every number from `eval/results/*.summary.json`. Do not type numbers by hand._

## Setup

- Corpus: _N_ unique HotpotQA dev paragraphs + 8 synthetic documents (1 duplicate skipped).
- Questions: 44 in `eval/questions.jsonl` (seed 42).

| Type | Count | Source | What a good system does |
|---|---|---|---|
| multi_hop | 15 + 3 | HotpotQA bridge + synthetic | finds both documents, answers correctly |
| comparison | 10 | HotpotQA comparison | finds both entities, compares |
| simple | 6 | synthetic | one document, exact fact |
| contradiction | 4 | synthetic | reports both values, cites both sources |
| temporal | 1 | synthetic | prefers the newer source, mentions the old one |
| unanswerable | 5 | synthetic | refuses (`insufficient_evidence`) |

- LLM: _model_, temperature 0. Embeddings: bge-small-en-v1.5.

## Metrics

| Metric | Definition |
|---|---|
| Recall@5, Recall@10 | share of relevant documents among the first K retrieved documents |
| MRR | mean of 1 / rank of the first relevant document |
| contains_ref | the normalized reference answer appears in the answer |
| F1 | word-overlap F1 with the reference (HotpotQA style) |
| judge | LLM grade 0 / 0.5 / 1 against the reference |
| citation_acc | share of cited documents that are relevant |
| abstention_correct | refused exactly when there was no answer |
| contradiction_found | at least one contradiction reported where one was planted |

## Comparison 1: vector-only vs hybrid (paragraph chunks)

_paste `python -m eval.run_eval compare vector_paragraph hybrid_paragraph`_

## Comparison 2: chunking A (fixed 300/50) vs B (paragraph 800)

_paste `python -m eval.run_eval compare hybrid_fixed hybrid_paragraph`_

## Full pipeline (hybrid, paragraph, with LLM)

_paste summary and `by_type`_

## Error analysis

For 5–10 failed questions: what was expected, what came out, which step failed (retrieval, plan, generation, validation), what would fix it.

| id | type | expected | got | failed step | fix |
|---|---|---|---|---|---|
| | | | | | |

## Conclusions and limits

_3–5 sentences: what works, what does not, what the numbers do not prove (44 questions, one model, one run)._

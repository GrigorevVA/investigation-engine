# Evaluation Report

Every number below comes from `eval/results/*.summary.json` (per-question rows in the matching `.jsonl`),
produced by `eval/run_eval.py` on 2026-10-27. Commands to reproduce are at the end.

## Setup

- **Corpus:** 66,581 unique HotpotQA dev (distractor) paragraphs + 8 synthetic documents in 4 formats
  (TXT, MD, PDF, DOCX; a 9th file is a planted duplicate and is skipped). 78,006 chunks with paragraph chunking.
- **Questions:** 44 in `eval/questions.jsonl` (HotpotQA picks fixed by seed 42 + 19 hand-written).
- **Embeddings:** BAAI/bge-small-en-v1.5 (fastembed, local). **Vector DB:** Qdrant Cloud (free) for the full runs.
- **LLMs (Groq free tier):** `qwen/qwen3.8-27b` and `openai/gpt-oss-120b`, temperature 0.
  `llama-3.3-70b-versatile`, the original plan, no longer exists on Groq (ADR 0007).

| Type | Count | Source | What a good system does |
|---|---|---|---|
| multi_hop | 15 + 3 | HotpotQA bridge + synthetic | finds both documents, answers correctly |
| comparison | 10 | HotpotQA comparison | finds both entities, compares |
| simple | 6 | synthetic | one document, exact fact |
| contradiction | 4 | synthetic (documents disagree on purpose) | reports both values, cites both sources |
| temporal | 1 | synthetic | prefers the newer source, mentions the old one |
| unanswerable | 5 | synthetic | refuses (`insufficient_evidence`) |

## Metrics

| Metric | Definition |
|---|---|
| Recall@5, Recall@10 | share of relevant documents among the first K retrieved documents |
| MRR | mean of 1 / rank of the first relevant document |
| contains_ref | the normalized reference answer appears in the answer |
| F1 | word-overlap F1 with the reference (HotpotQA style) |
| judge | LLM grade 0 / 0.5 / 1 against the reference (the same model that answered) |
| citation_acc | share of cited documents that are relevant (precision of citations) |
| abstention_correct | refused exactly when there was no answer |
| contradiction_found | at least one contradiction reported where one was planted |

## Comparison 1: vector vs BM25 vs hybrid (full corpus, retrieval only, one query = the question)

| metric | vector | BM25 | hybrid (RRF) |
|---|---|---|---|
| Recall@5 | 0.821 | 0.833 | **0.872** |
| Recall@10 | 0.885 | 0.872 | **0.923** |
| MRR | **0.966** | 0.927 | 0.946 |

Hybrid finds more of the relevant documents than either retriever alone (+3.8 points Recall@10 over vector):
RRF promotes chunks that both retrievers agree on, and BM25 catches exact names that the small embedding model
blurs. Vector-only has a slightly higher MRR; the 0.02 gap is about one question out of 44, so no conclusion.

All 6 hybrid misses are multi-hop questions where exactly one of the two documents was found: the second document
is never named in the question (e.g. "the organization Titulescu presided" = League of Nations). One query cannot
find it; that is the job of the planner and the search tool (Full pipeline below).

## Comparison 2: chunking A (fixed 300/50) vs B (paragraph, max 800)

Run on a reproducible subset: the first 10,000 corpus paragraphs + every document an eval question needs
(10,036 documents, `scripts/make_chunking_subset.py`). Embedding the full corpus with 300-character chunks
would take ~4 hours on the laptop used.

| metric | vector, fixed | vector, paragraph | hybrid, fixed | hybrid, paragraph |
|---|---|---|---|---|
| Recall@5 | 0.923 | 0.923 | 0.910 | 0.910 |
| Recall@10 | 0.936 | 0.936 | 0.962 | 0.949 |
| MRR | 0.978 | 0.979 | 0.978 | 0.979 |
| chunks stored | 24,969 | 11,922 | 24,969 | 11,922 |

Quality is the same within one question. HotpotQA paragraphs are short (637 characters on average), so the
paragraph strategy keeps one paragraph = one chunk, and fixed windows only split it into 2–3 pieces of the same
document. Fixed chunking stores 2.1x more chunks: twice the embedding time, storage and BM25 memory, for no gain.
**Decision: paragraph chunking.** On long reports with many topics per page the result could differ.

## Comparison 3: full pipeline, two LLMs (hybrid, paragraph, Qdrant Cloud)

| metric | qwen3.8-27b | gpt-oss-120b |
|---|---|---|
| Recall@5 | **0.908** | 0.897 |
| Recall@10 | 0.921 | 0.923 |
| MRR | 0.934 | 0.936 |
| contains_ref | **0.737** | 0.718 |
| F1 | 0.141 | **0.380** |
| judge | **0.837** | 0.784 |
| citation_acc | 0.838 | 0.833 |
| abstention_correct | 0.884 | 0.886 |
| contradiction_found | 1.0 | 1.0 |
| errors (429 after retries) | 1 | 0 |

By question type (qwen): simple 6/6 with the reference in the answer, unanswerable 5/5 refused,
contradiction 4/4 reported, comparison abstention_correct 0.7 (3 unnecessary refusals), multi-hop
contains_ref 0.722.

- **Planner + search tool help retrieval:** Recall@5 rises from 0.872 (one query) to 0.908.
- **Qwen is the default** (ADR 0009): best judge score and contains_ref, and its tool calls are valid.
  gpt-oss-120b on Groq nearly always sent its final answer as a call to a non-existent tool `json`
  (HTTP 400 `tool_use_failed`); the answer step catches that and asks again without tools, so it never used
  the search tool. Its higher F1 only reflects shorter answers: F1 punishes every extra word of a full sentence.
- contains_ref is 0 for contradiction and temporal questions by design: their references describe both values in a
  sentence that no answer repeats word for word; contradiction_found and the judge cover them.

## Error analysis (qwen, 8 imperfect answers out of 43)

| id | type | expected | got | failed step | fix |
|---|---|---|---|---|---|
| 5ae694a1 | multi_hop | defence of the Cathedral | "no information about Minister Pool" | retrieval: typo "Minister" for "Minster" | spelling-tolerant BM25 (character n-grams) or query correction by the planner |
| 5ac3090d | multi_hop | Bulgarian | Mexican and American sopranos | retrieval: the right soprano never named | more planner hops: list all laureates, then search each |
| 5a80caac | comparison | no | refused | retrieval: neither entity found (ranked under other Hall of Fame pages) | search each entity separately (the planner did not split it) |
| 5ac2659b | multi_hop | 25 laps | refused, though it quoted "10,000 m = 25 laps" | generation: too strict | allow one explicit inference step in the prompt |
| 5a8841ee | comparison | Sheridan County | refused: no coordinates | generation: needs outside knowledge (Montana vs India) | correct under the "documents only" rule; question is not answerable from the text |
| 5ac41b9e | comparison | flowers | described both genera, refused | generation + vague reference | judge-friendly reference; prompt to answer the shared category |
| 5ac3bab1 | multi_hop | Ernestine | Saxe-Coburg and Gotha | reference granularity | answer is half right (judge 0.5); multi-hop chain stopped one step early |
| 5ae7edee | multi_hop | Macaronesia | Canary Islands, Spain | generation stopped one hop early | ask the model to check the question's last relation |
| c1 | contradiction | both founding years | HTTP 429 after 5 retries | provider rate limit (8,000 tokens/min free) | longer pause between questions, or a paid tier |

Patterns: 3 of 8 are retrieval misses where the missing document is never named in the question;
3 are over-cautious refusals caused by the strict "only from the documents" rule (the price of zero invented facts:
no unanswerable question was answered); 2 are references at a different level of detail than the answer.

## Conclusions and limits

The system retrieves 92% of the needed documents in the top 10, cites relevant documents 84% of the time,
refuses all 5 unanswerable questions and reports all 4 planted contradictions. The main loss is multi-hop
questions whose second entity is not in the question, and over-strict refusals. Limits: 44 questions, one run per
configuration, the LLM judge is the same model that answered (self-grading bias), HotpotQA is English Wikipedia,
and the free LLM tier rate-limits a full run to tens of minutes.

## Reproduce

```bash
python -m eval.run_eval build
python -m eval.run_eval run --name vector_paragraph --collection documents_paragraph --mode vector --retrieval-only
python -m eval.run_eval run --name bm25_paragraph --collection documents_paragraph --mode bm25 --retrieval-only
python -m eval.run_eval run --name hybrid_paragraph --collection documents_paragraph --mode hybrid --retrieval-only
python -m scripts.make_chunking_subset --size 10000
QDRANT_URL= QDRANT_PATH=data/qdrant_subset python -m scripts.index_dataset --corpus data/corpus_subset.jsonl --strategy paragraph --collection subset_paragraph
QDRANT_URL= QDRANT_PATH=data/qdrant_subset python -m scripts.index_dataset --corpus data/corpus_subset.jsonl --strategy fixed --size 300 --overlap 50 --collection subset_fixed
QDRANT_URL= QDRANT_PATH=data/qdrant_subset python -m eval.run_eval run --name subset_hybrid_fixed --collection subset_fixed --mode hybrid --retrieval-only
LLM_MODEL=qwen/qwen3.8-27b python -m eval.run_eval run --name full_qwen --collection documents_paragraph --mode hybrid --judge --sleep 4
LLM_MODEL=openai/gpt-oss-120b python -m eval.run_eval run --name full_gptoss --collection documents_paragraph --mode hybrid --judge --sleep 4
python -m eval.run_eval compare full_qwen full_gptoss
```

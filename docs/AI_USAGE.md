# AI Usage Report

How AI tools were used to build this project, what they produced, and what was done by hand.
Update after every notable step.

## Tools

| Tool | Used for |
|---|---|
| Claude (claude.ai, agent mode) | Planning, project skeleton, ingestion, retrieval wrapper, RAG loop, API, UI, tests, docs drafts |
| _add others you use (Copilot, Cursor…)_ | |

## Rules we followed

- One module at a time: explanation, code, line-by-line review, then defence questions (see `CLAUDE.md`).
- The plan was for the author to write the own algorithms (BM25, Reciprocal Rank Fusion) and the retrieval metrics
  (Recall@K, MRR, citation accuracy) against failing spec tests. Because of the deadline the author changed this:
  the AI wrote them, the author studied them line by line to be able to rewrite them without help.
- Every number in reports comes from `eval/run_eval.py`, not from AI text.
- "Done" from the agent is not trusted: each requirement was checked against the code and a passing test.

## Log

| Date | Asked for | AI produced | Changed or written by the author |
|---|---|---|---|
| 2026-10-23 | Read the task, plan for 6 days | Plan, stack, rules (`CLAUDE.md`), day-by-day checklist | Chose HotpotQA and free tiers |
| 2026-10-27 | Project skeleton | Loaders, chunking, dedup, pipeline, Qdrant wrapper, LLM client with retries, planner, answer loop with tool calling, FastAPI, Gradio UI, scripts, 72 tests (17 of them a spec for the author's algorithms), ADRs | _fill in_ |
| 2026-10-27 | Implement BM25, RRF, metrics (author's decision to delegate) | `BM25Index` with an inverted index, `reciprocal_rank_fusion`, `recall_at_k`, `reciprocal_rank`, `citation_accuracy`; all 17 spec tests green | _author: line-by-line review, answers to the defence questions_ |

| 2026-10-27 | Get the data and index it | Switched the download to the Hugging Face copy (the CMU mirror no longer resolves); measured embedding speed (5–7 docs/s on an i5-10210U) and added length-sorted batching (12.5 docs/s); reproducible 10k subset for the chunking comparison; local → Qdrant Cloud copy script | _fill in_ |
| 2026-10-27 | Evaluate and pick a model | Retrieval runs (vector / BM25 / hybrid, two chunkings), full runs on two Groq models, error analysis and `docs/EVALUATION.md` from the result files; found and handled Groq `tool_use_failed`; ADR 0007, 0009 | _fill in_ |
| 2026-10-27 | Deploy for free | HF Docker Spaces turned out to need PRO (402). Measured memory (~625 MB) and cut it to a ~407 MB peak for Render's 512 MB: numpy CSR BM25, no texts in memory, HTML page instead of Gradio; `render.yaml`; ADR 0008. Live at https://investigation-engine-l74q.onrender.com | Author created the Render, Qdrant, Groq accounts and entered the secrets |

## Where AI was wrong or needed correction

- The first fixed-size chunker started overlaps in the middle of a word; caught by `test_fixed_does_not_cut_words`, fixed.
- The planned model `llama-3.3-70b-versatile` no longer existed on Groq; the AI had to check `GET /models` instead of trusting its plan.
- A local eval run silently went to Qdrant Cloud after `.env` was filled and created two empty collections; the AI deleted them and added a guard that refuses a missing collection.
- The first Render deploy crashed: `lru_cache` did not stop a request from building a second engine while the boot thread was building one (two BM25 indexes in 512 MB). Fixed with a lock and 503 while loading.
- `.gitignore` missed the subset Qdrant folder and logs; one log file got committed and was removed.
- _add your own findings_

## What I would not delegate next time

_fill in after the defence preparation_

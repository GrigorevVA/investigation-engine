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
- The own algorithms (BM25, Reciprocal Rank Fusion) and the retrieval metrics (Recall@K, MRR, citation accuracy)
  were written by the author. The AI wrote only their specification as failing tests and the formula in the docstring.
- Every number in reports comes from `eval/run_eval.py`, not from AI text.
- "Done" from the agent is not trusted: each requirement was checked against the code and a passing test.

## Log

| Date | Asked for | AI produced | Changed or written by the author |
|---|---|---|---|
| 2026-10-23 | Read the task, plan for 6 days | Plan, stack, rules (`CLAUDE.md`), day-by-day checklist | Chose HotpotQA and free tiers |
| 2026-10-27 | Project skeleton | Loaders, chunking, dedup, pipeline, Qdrant wrapper, LLM client with retries, planner, answer loop with tool calling, FastAPI, Gradio UI, scripts, 72 tests (17 of them a spec for the author's algorithms), ADRs | _fill in_ |
| | BM25, RRF, metrics | Not written by AI | _author: implemented `app/search/bm25.py`, `reciprocal_rank_fusion`, `eval/metrics.py`_ |

## Where AI was wrong or needed correction

- The first fixed-size chunker started overlaps in the middle of a word; caught by `test_fixed_does_not_cut_words`, fixed.
- _add your own findings_

## What I would not delegate next time

_fill in after the defence preparation_

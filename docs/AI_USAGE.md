# AI Usage Report

How AI tools were used to build this project, what they produced, and what was done by hand.
Update after every notable step.

## Tools

| Tool | Used for |
|---|---|
| Claude (claude.ai, agent mode) | Reading the task, 6-day plan, stack and rules (`CLAUDE.md`), first project skeleton |
| Claude Code (VS Code extension, model Claude Opus 5.5) | Everything from 2026-10-27: algorithms, data and indexing, evaluation runs and report, memory work, deployment, docs |

No other AI tools (Copilot, Cursor, ...) were used.

## How the work was split

- **AI wrote all of the code**, the tests, the evaluation runs and the first drafts of every document.
- **The author** set the goals and constraints (free infrastructure only, no LangChain or LlamaIndex, numbers only
  from `eval/run_eval.py`), chose the dataset, made the decisions the AI asked about (delegating the algorithms,
  Render as the host), created and owns every account (GitHub, Qdrant Cloud, Groq, Hugging Face, Render),
  and entered every secret directly into `.env` and the Render dashboard. The AI never saw the Groq or Qdrant keys.

## Rules we followed

- One module at a time, with an explanation and defence questions (see `CLAUDE.md`).
- The plan was for the author to write the own algorithms (BM25, Reciprocal Rank Fusion) and the retrieval metrics
  (Recall@K, MRR, citation accuracy) against failing spec tests. Because of the deadline the author changed this
  on 2026-10-27: the AI wrote them. The author's line-by-line study of this code for the defence is the next step.
- Every number in reports comes from `eval/run_eval.py`, not from AI text.
- "Done" from the agent is not trusted: each change landed with a passing test (86 tests at the end of 2026-10-27),
  and the live service was checked with real questions through the public URL.
- Small commits, one per step, all with the AI marked as co-author in the git history.

## Log

| Date | Asked for | AI produced | Decided or done by the author |
|---|---|---|---|
| 2026-10-23 | Read the task, plan for 6 days | Plan, stack, rules (`CLAUDE.md`), day-by-day checklist | Chose HotpotQA and free tiers |
| 2026-10-27 | Project skeleton | Loaders, chunking, dedup, pipeline, Qdrant wrapper, LLM client with retries, planner, answer loop with tool calling, FastAPI, Gradio UI, scripts, 72 tests (17 of them a spec for the author's algorithms), ADRs | Accepted the skeleton; no manual code changes |
| 2026-10-27 | Implement BM25, RRF, metrics | `BM25Index` with an inverted index, `reciprocal_rank_fusion`, `recall_at_k`, `reciprocal_rank`, `citation_accuracy`; all 17 spec tests green | Decided to delegate this to the AI because of the deadline (originally reserved for the author). Line-by-line study before the defence: pending |
| 2026-10-27 | Get the data and index it | Switched the download to the Hugging Face copy (the CMU mirror no longer resolves); measured embedding speed (5–7 docs/s on an i5-10210U) and added length-sorted batching (12.5 docs/s); reproducible 10k subset for the chunking comparison; local → Qdrant Cloud copy script | Created the GitHub repository, the Qdrant Cloud cluster and the Groq key; filled `.env` |
| 2026-10-27 | Evaluate and pick a model | Retrieval runs (vector / BM25 / hybrid, two chunkings), full runs on two Groq models, error analysis and `docs/EVALUATION.md` from the result files; found and handled Groq `tool_use_failed`; ADR 0007, 0009 | Accepted Qwen as the default on the evaluation evidence |
| 2026-10-27 | Deploy for free | HF Docker Spaces turned out to need PRO (402). Measured memory (~625 MB) and cut it to a ~407 MB peak for Render's 512 MB: numpy CSR BM25, no texts in memory, HTML page instead of Gradio; `render.yaml`; ADR 0008. Live at https://investigation-engine-l74q.onrender.com | Chose Render over Oracle Cloud and Koyeb from the options the AI laid out; created the Render service from the Blueprint and entered the secrets |

## Where AI was wrong or needed correction

- The first fixed-size chunker started overlaps in the middle of a word; caught by `test_fixed_does_not_cut_words`, fixed.
- The planned model `llama-3.3-70b-versatile` no longer existed on Groq; the AI had to check `GET /models` instead of trusting its plan.
- The dataset URL in the AI's own download script was dead (the CMU mirror no longer resolves).
- A local eval run silently went to Qdrant Cloud after `.env` was filled and created two empty collections; the AI deleted them and added a guard that refuses a missing collection.
- The AI's first memory plan was not enough: removing texts made the peak worse until the build-time copies were removed and Gradio was dropped. Measuring each step, not estimating, found the real costs.
- The first Render deploy crashed: `lru_cache` did not stop a request from building a second engine while the boot thread was building one (two BM25 indexes in 512 MB). Fixed with a lock and 503 while loading.
- `.gitignore` missed the subset Qdrant folder and logs; one log file got committed and was removed.

## Security incident

A Hugging Face write token was pasted into the AI chat. It was used once to try the Space deployment
(which failed with 402) and then revoked by the author on 2026-10-27. Lesson: tokens go into the terminal or the
hosting dashboard, never into a chat.

## What I would not delegate next time

_Draft from the facts above, to be rewritten by the author after the defence preparation:_

- The own algorithms (BM25, RRF, metrics). They are the part the defence asks about most, and writing them is
  faster than reverse-engineering someone else's version under time pressure.
- Choosing the hosting before checking its current free-tier limits; the HF Spaces change cost a redesign late in the project.

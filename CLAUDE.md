# CLAUDE.md — AI Investigation Engine

Rules for the AI assistant in this repository. Read before every task.

## Context
- Take-home task for the Upgaming Talent Growth Program, Applied AI. Due 29 October 2026.
- The author defends the code at an interview WITHOUT AI. Everything you write, the author must understand and be able to rewrite.
- The author is a QA engineer with basic Python. Explain in Russian, simply, with analogies (the repository itself is in English).

## How to work
1. One module at a time. Order: what and why → code → line-by-line walkthrough → 3 "what happens if..." questions, as at the defence.
2. BM25, RRF and the evaluation metrics (Recall@K, MRR, Citation Accuracy) were originally the author's to write. On 2026-10-27 the author delegated them to the AI because of the deadline. Every such piece of code now needs a line-by-line walkthrough and defence questions.
3. Do not make things up. Check library versions, tier limits and APIs, or mark them as an open question.
4. After every notable step add an entry to docs/AI_USAGE.md: what was asked, what the AI produced, what the author changed.
5. Do not add bonus features until the whole mandatory checklist is closed.
6. Before changing the architecture read docs/decisions. Record a new decision as a new ADR; never rewrite old ones.
7. "Done" is proven by a diff and a green test for each requirement in the README (Requirements coverage table), not by an agent's report.
8. Current state: everything mandatory is implemented, deployed (Render) and documented; pytest is green. Next: defence preparation with docs/WALKTHROUGH.md.

## Stack
Python 3.11 · FastAPI + an HTML page (no Gradio, ADR 0008) · Qdrant Cloud · bge-small via fastembed (local) · any OpenAI-compatible LLM (Groq, qwen/qwen3.8-27b, ADR 0009) · pytest · Docker · Render free (512 MB).
No LangChain or LlamaIndex: the whole RAG flow is visible in our own code.

## Code
- Type hints everywhere, functions under 40 lines, one responsibility per module.
- Layout: app/ingest, app/search, app/rag, app/api, app/ui, scripts, eval, tests, docs.
- All prompts in app/rag/prompts.py, all settings in app/config.py from .env.
- Secrets only in .env and the hosting secrets. Never in code or git.
- External service errors (LLM, Qdrant) are handled in llm.py and vector_store.py: timeout, retry with exponential backoff, clear message to the caller.
- The LLM answer is always structured JSON: answer, citations, contradictions, insufficient_evidence.
- Every module has a test in tests/. External APIs are mocked in tests.
- Every number in reports comes from a run of eval/run_eval.py.
- Small commits, one per task.

## Check before "done"
- pytest is green.
- The code runs with the command from the README.
- The author can answer the 3 check questions on the module without hints.

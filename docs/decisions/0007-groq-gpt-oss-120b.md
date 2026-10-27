# 0007. Groq `openai/gpt-oss-120b` as the default model

**Context.** On 2026-10-27 the planned default `llama-3.3-70b-versatile` returned `model_not_found` on Groq.
`GET /models` listed `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` as the general chat models.
Both `gpt-oss-120b` and `qwen3.8-27b` passed a check of the two features the pipeline needs:
JSON mode (`response_format: json_object`) and tool calling (`search_documents`).

**Decision.** `openai/gpt-oss-120b` is the default (the larger model). `qwen/qwen3.8-27b` is the second model
for the comparison run. Only `LLM_MODEL` changes; ADR 0005 stays valid.

**Limits seen in response headers (free tier).** 1000 requests, 8000 tokens per minute.
A daily token limit is not exposed in headers: open question, watch for 429 during the evaluation.
The client already waits on 429 using Retry-After (ADR 0005), and the eval sleeps between questions.

**Consequences.** One investigation is about 3–5 LLM calls and a few thousand tokens, so the 44-question run takes
tens of minutes on the free tier. Model names may change again: re-check `GET /models` before a demo.

# 0009. `qwen/qwen3.8-27b` as the default model

**Context.** ADR 0007 picked `openai/gpt-oss-120b` before any evaluation. The full 44-question run
(`eval/results/full_qwen.summary.json`, `full_gptoss.summary.json`) compared both on Groq.

**Evidence.** Judge 0.837 vs 0.784, contains_ref 0.737 vs 0.718, Recall@5 0.908 vs 0.897; citation accuracy and
abstention equal. gpt-oss-120b nearly always returned its final answer as a call to a non-existent tool
(`tool_use_failed`), so it never used the search tool; Qwen's tool calls were valid.

**Decision.** Default `LLM_MODEL=qwen/qwen3.8-27b` (config, `.env.example`, `render.yaml`).
Supersedes the model choice in ADR 0007; its notes on limits still hold.

**Consequences.** Qwen writes longer answers (lower word-overlap F1, 0.141 vs 0.380), which does not hurt the judge
or contains_ref. Model names on Groq change; re-check `GET /models` before a demo.

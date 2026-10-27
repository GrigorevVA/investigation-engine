# 0005. OpenAI-compatible LLM endpoint

**Context.** Free tiers: Groq and Google Gemini. Their limits and model names change.

**Decision.** One client for the `/chat/completions` format, configured by `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. Temperature 0.

**Consequences.** Switching provider is a config change. Errors: 408/429/5xx and timeouts are retried with exponential backoff (honouring Retry-After), then reported as 503; other 4xx fail at once as 502.

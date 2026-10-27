# 0002. Own RAG loop, no framework

**Context.** LangChain or LlamaIndex would give retrieval chains, tool calling and loaders ready-made.

**Decision.** Plain Python: httpx for the LLM, qdrant-client, small modules we own.

**Why.** The defence is without AI help: every step of the flow must be visible in our code and explainable. Frameworks hide prompts and retries and add version churn.

**Consequences.** More code to write and test (loaders, retries, tool loop). Each module stays under ~150 lines.

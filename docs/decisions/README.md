# Decision log

One file per decision (context, options, decision, consequences). Records are never rewritten:
a changed decision gets a new record that supersedes the old one. Read these before changing the design.

| # | Decision | Status |
|---|---|---|
| [0001](0001-dataset-hotpotqa.md) | HotpotQA dev distractor paragraphs as the 50k+ corpus | accepted |
| [0002](0002-no-rag-framework.md) | Own RAG loop, no LangChain or LlamaIndex | accepted |
| [0003](0003-local-embeddings-qdrant-cloud.md) | Local bge-small embeddings, Qdrant Cloud for persistence | accepted |
| [0004](0004-hybrid-bm25-rrf.md) | Hybrid retrieval: own BM25 + own RRF | accepted |
| [0005](0005-openai-compatible-llm.md) | Any OpenAI-compatible LLM, chosen by config | accepted |
| [0006](0006-citation-labels.md) | Short chunk labels and server-side citation check | accepted |
| [0007](0007-groq-gpt-oss-120b.md) | Groq `openai/gpt-oss-120b` as the default model | accepted |

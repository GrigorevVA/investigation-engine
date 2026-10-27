# 0004. Hybrid retrieval with own BM25 and own RRF

**Context.** Semantic search misses exact names, dates and codes; keyword search misses paraphrases. The task asks for at least one own retrieval or ranking algorithm.

**Decision.** BM25 (Okapi, k1 = 1.5, b = 0.75) over title + text, in memory; Reciprocal Rank Fusion (k = 60) of the top 30 from each retriever.

**Why RRF.** Cosine scores and BM25 scores live on different scales; RRF uses ranks only, so no score normalisation is needed.

**Consequences.** BM25 is rebuilt from the Qdrant payloads at start and after each upload or delete. The evaluation compares vector-only and hybrid.

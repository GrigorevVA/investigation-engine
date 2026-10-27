# 0003. Local embeddings, Qdrant Cloud

**Context.** 60k+ chunks must be embedded; the hosted app must survive restarts; budget is zero.

**Decision.** `BAAI/bge-small-en-v1.5` (384 dimensions) through fastembed (ONNX, CPU), run on the laptop once. Vectors and payloads live in Qdrant Cloud free tier. Locally, an empty `QDRANT_URL` switches to on-disk Qdrant.

**Why.** No embedding API costs or rate limits; documents are not sent anywhere for indexing. The hosting disk is ephemeral, a managed database is not.

**Consequences.** The Space downloads the same model at build time to embed queries. Open question: confirm the current free-tier limits of Qdrant Cloud fit two collections.

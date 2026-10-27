# 0008. Render free web service; fit the app in 512 MB

**Context.** On 2026-10-27 creating a Docker Space on Hugging Face returned
`402 Payment Required: hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription`.
The task requires free infrastructure. Free web services on Render and Koyeb give 512 MB of RAM.
Measured on the full collection (78,006 chunks), the app needed ~625 MB:
bge-small model ~190 MB, chunk texts ~90 MB, BM25 as Python lists ~250 MB, Gradio ~120 MB.

**Options.** Oracle Cloud Always Free (enough RAM, but card verification and manual server setup);
Hugging Face PRO (paid, against the task); Render or Koyeb free with a smaller app.

**Decision.** Render free web service (Docker, `render.yaml`) and a smaller app:
- BM25 postings as numpy CSR arrays instead of Python tuples (~250 MB -> ~40 MB, same scores, tests unchanged);
- no chunk texts in memory: BM25 keeps `(chunk_id, doc_id)`, texts of the hits are fetched from Qdrant by id;
- the document registry lists only uploaded and synthetic documents; corpus paragraphs are counted;
- after an upload or delete BM25 is rebuilt from Qdrant in a background thread; the old index is freed first,
  so during the rebuild (~30 s locally) hybrid search falls back to vector search;
- a plain HTML + JS page over the REST API replaces Gradio.

**Consequences.** Measured peak 407 MB locally (Windows), ~100 MB headroom. Supersedes the hosting part of
ADR 0003 (Qdrant Cloud stays). The free service sleeps when idle and has a fraction of a CPU, so the first
request after sleep waits for the start and the BM25 build (minutes). `/api/ping` answers at once for the
platform health check while the index loads in the background.

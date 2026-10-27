# Project walkthrough for the defence

Read top to bottom, one module at a time. After each module close the file and answer the questions out loud
without looking.

## The path of one question

The user types a question on the page `app/ui/index.html`. The page sends it to `/api/ask`, which calls
`Engine.investigate` (`app/engine.py`). That creates an `Investigator` (`app/rag/answer.py`). The planner
(`app/rag/planner.py`) asks the LLM to split the question into search queries. For each query `HybridSearcher`
(`app/search/hybrid.py`) searches Qdrant by vector and BM25 by words, and merges the two lists with RRF.
The chunks found get labels C1, C2, ... and go to the LLM together with the question. The LLM either asks for
another search through the `search_documents` tool, or returns JSON with the answer, citation labels,
contradictions and an insufficient-evidence flag. The server drops labels it never showed and returns the result
to the page.

## 1. Loading (`app/ingest/loaders.py`)

Each format is read by its own function: PDF page by page with pypdf, DOCX paragraph by paragraph with
python-docx, TXT and MD as text. Then the text is normalized (same line endings, at most one empty line in a row).
`doc_id` is built from the file name, so uploading a file with the same name again counts as an update.

Questions. Why is `doc_id` built from the name and not from the content? What happens with a scanned PDF without a
text layer? Where is the metadata stored and why?

Answers. From the content alone a "new version" could not be told apart from a "new document". A scan gives empty
text, and the loader refuses with a clear error; OCR is out of scope. Metadata sits in the Qdrant payload of every
chunk and is shown in the interface.

## 2. Chunking (`app/ingest/chunking.py`)

Strategy A cuts the text into fixed-size windows with overlap and does not break words. Strategy B keeps
paragraphs whole, merges short ones and cuts only oversized ones with strategy A. `title + text` goes into the
index and the embedding, because a short paragraph without its title loses meaning.

Questions. Why overlap? Why do the strategies barely differ on HotpotQA? Which is worse, a chunk that is too big
or one that is too small?

Answers. Overlap saves a fact that fell on the border of two windows. HotpotQA paragraphs average 637 characters,
so strategy B keeps one paragraph = one chunk and A at 300 characters only splits the same document into 2–3
pieces; recall is the same, but A stores 2.1x more chunks (`docs/EVALUATION.md`). A big chunk blurs the embedding
with several topics, a small one loses context; the choice is confirmed by metrics.

## 3. Duplicates, add, update, delete (`dedup.py`, `pipeline.py`)

The SHA-256 hash is taken from the text ignoring case and whitespace. The decision table is in the docstring of
`IngestPipeline.ingest`. Same `doc_id` and same hash means "unchanged". Same `doc_id` and a new hash means
update: all old chunks are deleted first, so no leftovers stay. A new `doc_id` with an already known hash means
duplicate.

Questions. Why delete all chunks of a document before an update? Why is the Qdrant point id a uuid5? What happens
when the same corpus is indexed again?

Answers. A new version may produce fewer chunks, and the extra old ones would stay in search. Qdrant accepts only
a UUID or an integer, and uuid5 of `chunk_id` always gives the same UUID. So re-indexing overwrites points and
does not create copies.

## 4. Vector search (`embeddings.py`, `vector_store.py`)

bge-small turns a text into a vector of 384 numbers locally, without an API. For queries fastembed adds the
instruction prefix the bge model expects. Texts are embedded in length order: a batch is padded to its longest
text, so sorting by length removes wasted compute (1.8x faster on the laptop). Qdrant finds the nearest vectors
by cosine. Every Qdrant call goes through `_call`, where network errors and 5xx are retried and everything else
becomes a clear `VectorStoreError`.

Questions. Why Qdrant Cloud and not a file on the host? What does a cosine of 0.9 mean? Why are the query and the
document embedded by different methods?

Answers. The free host's disk is wiped on restart. A cosine near 1 means the vectors point the same way: the texts
are close in meaning. bge was trained with an instruction for queries; without it search quality drops.

## 5. BM25, own algorithm (`app/search/bm25.py`)

The formula is in the docstring. The constructor goes over all chunks once: tokenizes `search_text`, counts each
word in the chunk (`Counter`) and the chunk length. It stores an inverted index "word → chunks that contain it"
in CSR form: three flat numpy arrays instead of millions of Python objects (~40 MB instead of ~250 MB, which is
what lets the app fit 512 MB). `doc_ids` and `tfs` hold the postings sorted by word id, and `offsets[t]` ..
`offsets[t+1]` is the slice of word `t`. The length part of the denominator depends only on the chunk, so it is
computed once (`norm`). `search` takes only the slices of the query words, adds their score contributions with
numpy, and takes the top k with a partial sort (`argpartition`). With `drop_text=True` the index keeps only
`(chunk_id, doc_id)`; texts of the hits are fetched from Qdrant later.

Questions. What does k1 do? What does b do? Why the `+ 1` inside the idf logarithm? Why an inverted index and
not a loop over all chunks?

Answers. k1 limits the effect of repeated words: the tenth repetition adds almost nothing. b penalizes long chunks,
otherwise they would win just by having more words. Without `+ 1` a word found in more than half of the chunks
would get a negative idf. The inverted index touches only the chunks that contain a query word, not all
78,000 (3 searches measured ~3 ms on the laptop).

## 6. RRF, own algorithm (`reciprocal_rank_fusion` in `app/search/hybrid.py`)

For each ranked list go through it with a position starting at 1 and add `1 / (k + position)` to the chunk's
score. Sort by score, highest first. Tests are in `tests/test_hybrid.py`.

Questions. Why not add up the raw scores of the two searches? Why k = 60? What does appearing in both lists give
a chunk?

Answers. Cosine (0 to 1) and BM25 (0 to tens) are not comparable. k smooths the gap between the top places, so
first place does not take everything. A chunk in both lists gets two contributions and moves up; this is why
hybrid Recall@10 is 0.923 against 0.885 for vector alone.

## 7. LLM and errors (`app/rag/llm.py`, `app/retry.py`)

One client for any OpenAI-compatible API. Timeouts, 408, 429 and 5xx are retried with a pause of 1, 2, 4, 8
seconds plus a small random jitter. If the server sent Retry-After, we wait at least that long. 401 or 400 are not
retried: the request is wrong and a retry will not help. A special case is Groq's 400 `tool_use_failed` (the model
called a tool that does not exist): it raises `ToolUseFailed`, and the answer step asks again without tools. After
all retries the API returns 503 to the client with a clear message.

Questions. Why random jitter in the pause? Why not retry 401? What does the user see when Groq is down?

Answers. So that many clients do not retry at the same moment. A wrong key will not fix itself. The message "the
language model is unavailable right now" instead of hanging or a traceback.

## 8. Investigation (`planner.py`, `answer.py`, `tools.py`, `prompts.py`)

The planner always puts the original question first, then the sub-questions. If the plan is broken, we search by
the question itself. The context is at most 12 labelled chunks. The model may use the search tool for 2 rounds;
on the last round no tools are passed, so it must answer. Then citations and flags are checked.

Questions. How does the system refuse to answer? How is an invented citation caught? How do we defend against
instructions inside documents?

Answers. The model sets `insufficient_evidence`, and the server sets it itself if no valid citation is left.
Labels are checked against the dictionary of chunks that were shown. The prompt says chunk text is data, not
instructions, and the synthetic data contains a document with such an attack to test it.

## 9. Metrics, own code (`eval/metrics.py`)

Recall@K is the share of relevant documents among the first K. Reciprocal rank is `1 / position` of the first
relevant document, or 0. Citation accuracy is the share of cited documents that are relevant. Tests are in
`tests/test_metrics.py`.

Questions. How is Recall@K different from MRR? Why count documents and not chunks? Why is citation accuracy a
precision and not a recall?

Answers. Recall says whether we found it at all, MRR says how high. HotpotQA labels are per article, and an
article may have several chunks. What matters is that every citation is honest; how complete the search is,
Recall already measures.

## 10. API, interface, deployment

FastAPI serves `/api/*` and the HTML page at the root, in one process. Hosting is the free Render web service with
512 MB of RAM (ADR 0008), so BM25 lives in numpy arrays and chunk texts are not kept in memory but fetched from
Qdrant. On start a background thread reads all chunks from Qdrant and builds BM25; until it is ready the API
answers 503 "the index is still loading" and `/api/ping` answers at once for the platform health check. The engine
is built exactly once under a lock: two concurrent builds did not fit in 512 MB and crashed the first deploy.
Keys live only in Render's secrets and the local `.env`.

Questions. What happens when the service restarts on Render? Where are the keys? Why are corpus embeddings not
computed on the host?

Answers. The data is in Qdrant Cloud and BM25 is rebuilt at start. In Render's secrets and in the local `.env`,
which is in `.gitignore`. The free CPU would take far too long to embed 66,000 paragraphs (the laptop did ~12
per second); it is done once on the laptop and the vectors are copied to the cloud.

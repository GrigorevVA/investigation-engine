# 0001. HotpotQA dev distractor as the corpus

**Context.** The task needs 50,000+ documents and questions we can grade. Options: HotpotQA, 2WikiMultihopQA, RAGBench, postmortems.

**Decision.** Unique paragraphs of HotpotQA dev (distractor), plus 9 synthetic documents in four formats.

**Why.** Every question lists its supporting paragraphs, so Recall@K, MRR and citation accuracy have ground truth for free. Questions are multi-hop and comparison by design. One 45 MB file, no account needed.

**Consequences.** Simple, unanswerable and contradiction questions are not in HotpotQA, so they come from the synthetic documents (`scripts/make_synthetic_docs.py`). The corpus is English. Paragraphs are short, so the two chunking strategies differ mostly on longer uploaded documents and with a small fixed size (300 characters).

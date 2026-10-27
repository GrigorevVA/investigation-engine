# 0006. Chunk labels and citation check

**Context.** Models invent sources. Long chunk ids in prompts are easy to garble.

**Decision.** Chunks are shown as `C1…Cn`. The model returns labels; the server keeps only labels it actually showed, maps them back to chunk ids, and marks an answer without valid citations as unsupported.

**Consequences.** Citation accuracy can be measured exactly. Invented citations are visible in the result `notes`.

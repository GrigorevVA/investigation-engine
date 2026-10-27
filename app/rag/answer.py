"""The investigation loop: plan -> retrieve -> answer with citations (+ optional tool calls)."""
import json
import logging
from dataclasses import asdict, dataclass, field

from app.models import Chunk, SearchHit
from app.rag.llm import LLMClient, parse_json
from app.rag.planner import plan_queries
from app.rag.prompts import ANSWER_SYSTEM, ANSWER_USER
from app.rag.tools import TOOLS, parse_tool_call
from app.search.hybrid import HybridSearcher

log = logging.getLogger(__name__)


@dataclass
class Citation:
    label: str
    chunk_id: str
    doc_id: str
    title: str
    snippet: str


@dataclass
class InvestigationResult:
    question: str
    answer: str
    insufficient_evidence: bool
    citations: list[Citation] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)  # plan + queries the LLM asked for via the tool
    retrieved: list[Citation] = field(default_factory=list)  # everything shown to the LLM, in order
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Context:
    """Chunks shown to the LLM, labelled C1, C2... in the order they were found."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.by_label: dict[str, Chunk] = {}
        self._seen: set[str] = set()

    def add(self, hits: list[SearchHit]) -> list[str]:
        new_labels = []
        for hit in hits:
            if len(self.by_label) >= self.limit or hit.chunk.chunk_id in self._seen:
                continue
            label = f"C{len(self.by_label) + 1}"
            self.by_label[label] = hit.chunk
            self._seen.add(hit.chunk.chunk_id)
            new_labels.append(label)
        return new_labels

    def render(self, labels: list[str]) -> str:
        return "\n\n".join(f"[{l}] (document: {self.by_label[l].title})\n{self.by_label[l].text}" for l in labels)

    def citation(self, label: str) -> Citation:
        c = self.by_label[label]
        return Citation(label, c.chunk_id, c.doc_id, c.title, c.text[:300])


class Investigator:
    def __init__(self, llm: LLMClient, searcher: HybridSearcher, top_k: int = 8, mode: str = "hybrid",
                 max_subquestions: int = 3, max_context_chunks: int = 12, max_tool_rounds: int = 2) -> None:
        self.llm = llm
        self.searcher = searcher
        self.top_k = top_k
        self.mode = mode
        self.max_subquestions = max_subquestions
        self.max_tool_rounds = max_tool_rounds
        self.max_context_chunks = max_context_chunks

    def investigate(self, question: str) -> InvestigationResult:
        queries = plan_queries(question, self.llm, self.max_subquestions)
        context = Context(self.max_context_chunks)
        per_query = max(2, self.top_k // len(queries)) if len(queries) > 1 else self.top_k
        for query in queries:
            context.add(self.searcher.search(query, per_query, self.mode))
        if not context.by_label:
            return InvestigationResult(question, "No relevant documents were found.", True, queries=queries)

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM.format(max_rounds=self.max_tool_rounds)},
            {"role": "user", "content": ANSWER_USER.format(
                question=question, plan="; ".join(queries), context=context.render(list(context.by_label)))},
        ]
        raw = self._answer_loop(messages, context, queries)
        return self._build_result(question, raw, context, queries)

    def _answer_loop(self, messages: list[dict], context: Context, queries: list[str]) -> dict:
        for round_no in range(self.max_tool_rounds + 1):
            allow_tools = round_no < self.max_tool_rounds
            message = self.llm.chat(messages, tools=TOOLS if allow_tools else None)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls or not allow_tools:
                return self._parse_answer(message, messages)
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": tool_calls})
            for call in tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": self._run_tool(call, context, queries)})
        raise AssertionError("unreachable")

    def _run_tool(self, call: dict, context: Context, queries: list[str]) -> str:
        name, args = parse_tool_call(call)
        query = str(args.get("query", "")).strip()
        if name != "search_documents" or not query:
            return json.dumps({"error": "unknown tool or empty query"})
        queries.append(query)
        labels = context.add(self.searcher.search(query, self.top_k, self.mode))
        return context.render(labels) if labels else "No new chunks found for this query."

    def _parse_answer(self, message: dict, messages: list[dict]) -> dict:
        try:
            return parse_json(message.get("content") or "")
        except ValueError:
            log.warning("Answer was not valid JSON, asking once more")
            retry = messages + [message, {"role": "user", "content": "Reply with only the JSON object described."}]
            return self.llm.chat_json(retry)

    def _build_result(self, question: str, raw: dict, context: Context, queries: list[str]) -> InvestigationResult:
        """Trust nothing the model says about sources: keep only labels we actually showed it."""
        notes: list[str] = []
        labels = _valid_labels(raw.get("citations", []), context, notes)
        contradictions = []
        for item in raw.get("contradictions") or []:
            if isinstance(item, dict) and item.get("description"):
                cited = _valid_labels(item.get("citations", []), context, notes)
                contradictions.append({"description": str(item["description"]), "citations": cited})
        insufficient = bool(raw.get("insufficient_evidence", False))
        if not insufficient and not labels:
            insufficient = True
            notes.append("The answer had no valid citations, so it is marked as not supported by the documents.")
        return InvestigationResult(
            question=question,
            answer=str(raw.get("answer", "")).strip() or "No answer was produced.",
            insufficient_evidence=insufficient,
            citations=[context.citation(l) for l in labels],
            contradictions=contradictions,
            queries=queries,
            retrieved=[context.citation(l) for l in context.by_label],
            notes=notes,
        )


def _valid_labels(raw_labels, context: Context, notes: list[str]) -> list[str]:
    labels: list[str] = []
    for item in raw_labels if isinstance(raw_labels, list) else []:
        label = str(item).strip("[] ")
        if label in context.by_label and label not in labels:
            labels.append(label)
        elif label not in context.by_label:
            notes.append(f"Dropped citation {label!r}: it was not in the context.")
    return labels

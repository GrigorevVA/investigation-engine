"""The investigation loop with a scripted fake LLM and a real in-memory store."""
import json

from app.ingest.pipeline import IngestPipeline
from app.rag.answer import Investigator
from app.search.hybrid import HybridSearcher
from tests.conftest import FakeLLM, make_doc


def plan(*queries: str) -> dict:
    return {"role": "assistant", "content": json.dumps({"subquestions": list(queries)})}


def final(answer: str, citations: list[str], contradictions=None, insufficient=False) -> dict:
    body = {"answer": answer, "citations": citations, "contradictions": contradictions or [],
            "insufficient_evidence": insufficient}
    return {"role": "assistant", "content": json.dumps(body)}


def tool_call(query: str) -> dict:
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "search_documents", "arguments": json.dumps({"query": query})}}]}


def setup(store, embedder, replies) -> tuple[Investigator, FakeLLM]:
    IngestPipeline(store, embedder, "paragraph", 800, 100).ingest_batch([
        make_doc("profile", "Lumora Robotics was founded in 2014 by Nino Beridze.", "Profile"),
        make_doc("press", "Lumora Robotics, founded in 2016, named David Kapanadze CEO.", "Press release"),
        make_doc("spec", "Atlas-3 payload is 12 kg.", "Spec"),
    ])
    llm = FakeLLM(replies)
    return Investigator(llm, HybridSearcher(store, embedder, bm25=None), top_k=3, mode="vector"), llm


def test_answer_with_valid_citations(store, embedder):
    inv, llm = setup(store, embedder, [plan("Lumora founded year"), final("Founded in 2014 [C1].", ["C1"])])
    result = inv.investigate("When was Lumora founded?")
    assert not result.insufficient_evidence
    assert [c.label for c in result.citations] == ["C1"]
    assert result.queries[0] == "When was Lumora founded?"
    assert "Lumora founded year" in result.queries
    assert llm.calls[1]["tools"]  # the answer step offers the search tool


def test_invented_citation_is_dropped_and_answer_flagged(store, embedder):
    inv, _ = setup(store, embedder, [plan(), final("Founded in 1999 [C42].", ["C42"])])
    result = inv.investigate("When was Lumora founded?")
    assert result.citations == []
    assert result.insufficient_evidence is True
    assert any("C42" in n for n in result.notes)


def test_tool_call_adds_new_context(store, embedder):
    inv, llm = setup(store, embedder, [plan(), tool_call("Atlas-3 payload kg"), final("12 kg [C2].", ["C2"])])
    inv.top_k = 1
    result = inv.investigate("Who founded Lumora?")
    assert "Atlas-3 payload kg" in result.queries
    tool_messages = [m for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert tool_messages and tool_messages[0]["tool_call_id"] == "call_1"


def test_tool_rounds_are_limited(store, embedder):
    replies = [plan(), tool_call("a"), tool_call("b"), final("Done [C1].", ["C1"])]
    inv, llm = setup(store, embedder, replies)
    inv.max_tool_rounds = 2
    inv.investigate("Question?")
    assert llm.calls[-1]["tools"] is None  # last round must answer, no more tools


def test_contradictions_are_kept_with_valid_labels(store, embedder):
    contradiction = [{"description": "2014 vs 2016", "citations": ["C1", "C2", "C9"]}]
    inv, _ = setup(store, embedder, [plan(), final("Sources disagree [C1][C2].", ["C1", "C2"], contradiction)])
    result = inv.investigate("When was Lumora founded?")
    assert result.contradictions == [{"description": "2014 vs 2016", "citations": ["C1", "C2"]}]


def test_model_refusal_is_respected(store, embedder):
    inv, _ = setup(store, embedder, [plan(), final("The documents do not state 2031 revenue.", [], insufficient=True)])
    assert inv.investigate("Revenue in 2031?").insufficient_evidence is True


def test_planner_garbage_falls_back_to_question(store, embedder):
    bad_plan = {"role": "assistant", "content": "not json at all"}
    inv, _ = setup(store, embedder, [bad_plan, bad_plan, final("Answer [C1].", ["C1"])])
    assert inv.investigate("When was Lumora founded?").queries == ["When was Lumora founded?"]


def test_empty_collection_refuses_without_answer_call(store, embedder):
    llm = FakeLLM([plan()])
    inv = Investigator(llm, HybridSearcher(store, embedder, bm25=None), mode="vector")
    result = inv.investigate("Anything?")
    assert result.insufficient_evidence and len(llm.calls) == 1

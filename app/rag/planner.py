"""Break a complex question into search queries."""
import logging

from app.rag.llm import LLMClient, LLMUnavailable
from app.rag.prompts import PLANNER_SYSTEM

log = logging.getLogger(__name__)


def plan_queries(question: str, llm: LLMClient, max_sub: int) -> list[str]:
    """The original question always comes first, then up to max_sub sub-queries.
    If planning fails for any reason except an outage, we still search with the question itself."""
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM.format(max_sub=max_sub)},
        {"role": "user", "content": question},
    ]
    try:
        raw = llm.chat_json(messages).get("subquestions", [])
    except LLMUnavailable:
        raise
    except Exception as err:  # broken JSON, odd answer shape: degrade, do not fail
        log.warning("Planner failed, searching with the question only: %s", err)
        raw = []
    queries = [question]
    for item in raw if isinstance(raw, list) else []:
        text = str(item).strip()
        if text and text.lower() not in {q.lower() for q in queries}:
            queries.append(text)
    return queries[: max_sub + 1]

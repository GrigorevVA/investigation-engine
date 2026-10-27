"""Tools the LLM may call (OpenAI function calling format)."""
import json

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the document collection for more evidence. Use a short keyword-rich query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to search for"}},
            "required": ["query"],
        },
    },
}

TOOLS = [SEARCH_TOOL]


def parse_tool_call(tool_call: dict) -> tuple[str, dict]:
    """Return (tool name, arguments). Arguments arrive as a JSON string."""
    function = tool_call.get("function", {})
    raw = function.get("arguments") or "{}"
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except json.JSONDecodeError:
        args = {}
    return function.get("name", ""), args

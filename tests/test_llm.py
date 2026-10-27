import httpx
import pytest

from app.rag.llm import LLMClient, LLMError, LLMUnavailable, parse_json
from app.retry import backoff_delay

OK = {"choices": [{"message": {"role": "assistant", "content": '{"a": 1}'}}]}


def client(handler, retries: int = 3) -> tuple[LLMClient, list[float]]:
    sleeps: list[float] = []
    c = LLMClient("https://llm.test/v1", "key", "model", max_retries=retries, backoff_base_s=1.0,
                  transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return c, sleeps


def test_retries_on_429_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "3"}) if len(calls) < 3 else httpx.Response(200, json=OK)

    c, sleeps = client(handler)
    assert c.chat([{"role": "user", "content": "hi"}])["content"] == '{"a": 1}'
    assert len(calls) == 3
    assert all(s >= 3 for s in sleeps)  # Retry-After is respected


def test_timeout_gives_up_with_llm_unavailable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    c, sleeps = client(handler, retries=2)
    with pytest.raises(LLMUnavailable):
        c.chat([{"role": "user", "content": "hi"}])
    assert len(sleeps) == 2


def test_bad_request_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="bad key")

    c, _ = client(handler)
    with pytest.raises(LLMError) as err:
        c.chat([{"role": "user", "content": "hi"}])
    assert not isinstance(err.value, LLMUnavailable)
    assert len(calls) == 1


def test_request_body_has_tools_and_json_mode():
    seen = {}

    def handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=OK)

    c, _ = client(handler)
    c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}], json_mode=True)
    assert seen["tools"] == [{"type": "function"}]
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["temperature"] == 0.0


def test_backoff_grows_exponentially():
    assert backoff_delay(0, 1.0) < backoff_delay(3, 1.0)
    assert backoff_delay(3, 1.0) >= 8


@pytest.mark.parametrize("text", ['{"x": 1}', '```json\n{"x": 1}\n```', 'Here you go: {"x": 1} done'])
def test_parse_json_tolerates_wrappers(text):
    assert parse_json(text) == {"x": 1}


def test_parse_json_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json("no json here")

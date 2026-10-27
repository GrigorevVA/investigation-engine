"""Client for any OpenAI-compatible chat API, with timeouts, retries and JSON parsing."""
import json
import re

import httpx

from app.config import Settings
from app.retry import RetryableError, with_retry

RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The request is wrong (bad key, bad model). Retrying will not help."""


class LLMUnavailable(LLMError):
    """The provider is down, slow or rate limiting us after all retries."""


class ToolUseFailed(LLMError):
    """The model produced a malformed tool call (Groq: 400 tool_use_failed).
    Seen with gpt-oss-120b: it 'calls' a tool named json to return its final answer."""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 30.0,
                 max_retries: int = 4, backoff_base_s: float = 1.0,
                 transport: httpx.BaseTransport | None = None, sleep=None) -> None:
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_settings(cls, s: Settings) -> "LLMClient":
        return cls(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_s, s.llm_max_retries, s.llm_backoff_base_s)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, temperature: float = 0.0) -> dict:
        """One chat completion. Returns the assistant message dict (content and/or tool_calls)."""
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        kwargs = {"sleep": self._sleep} if self._sleep else {}
        try:
            data = with_retry(lambda: self._post(body), self.max_retries, self.backoff_base_s, **kwargs)
        except RetryableError as err:
            raise LLMUnavailable(f"LLM is unavailable after {self.max_retries + 1} attempts: {err}") from err
        return data["choices"][0]["message"]

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        """Chat that must return a JSON object. One repair attempt if the model returns broken JSON."""
        message = self.chat(messages, json_mode=True, temperature=temperature)
        try:
            return parse_json(message.get("content") or "")
        except ValueError:
            repair = messages + [message, {"role": "user", "content": "Return only one valid JSON object."}]
            return parse_json(self.chat(repair, json_mode=True).get("content") or "")

    def _post(self, body: dict) -> dict:
        try:
            response = self.http.post("/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as err:
            raise RetryableError(f"{type(err).__name__}: {err}") from err
        if response.status_code in RETRY_STATUSES:
            raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
        if response.status_code == 400 and "tool_use_failed" in response.text:
            raise ToolUseFailed(f"HTTP 400: {response.text[:300]}")
        if response.status_code >= 400:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:300]}")
        return response.json()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def parse_json(text: str) -> dict:
    """Parse a JSON object, tolerating ```json fences and text around the object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in model output: {text[:200]!r}")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as err:
            raise ValueError(f"invalid JSON in model output: {err}") from err
    if not isinstance(value, dict):
        raise ValueError("model output is JSON but not an object")
    return value

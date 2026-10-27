"""Retry with exponential backoff, shared by the LLM client and the vector store."""
import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)


class RetryableError(Exception):
    """Raise this inside the wrapped call to ask for another attempt."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def backoff_delay(attempt: int, base_s: float, retry_after_s: float | None = None) -> float:
    """1s, 2s, 4s, 8s... plus a little jitter. A server's Retry-After wins if it is longer."""
    delay = base_s * (2**attempt) + random.uniform(0, base_s / 2)
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    return delay


def with_retry(
    call: Callable[[], T],
    max_retries: int,
    base_s: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run call(); on RetryableError wait and try again, at most max_retries extra times."""
    for attempt in range(max_retries + 1):
        try:
            return call()
        except RetryableError as err:
            if attempt == max_retries:
                raise
            delay = backoff_delay(attempt, base_s, err.retry_after_s)
            log.warning("Attempt %d failed (%s), retrying in %.1fs", attempt + 1, err, delay)
            sleep(delay)
    raise AssertionError("unreachable")

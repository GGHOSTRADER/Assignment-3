# llm_utils.py
import random
import time
from typing import Any


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect common rate-limit / quota errors by message content.
    Keeps this provider-agnostic.
    """
    msg = str(exc).lower()

    rate_limit_signals = [
        "429",
        "rate limit",
        "quota exceeded",
        "too many requests",
        "resource exhausted",
        "retry in",
        "requests per minute",
    ]
    return any(signal in msg for signal in rate_limit_signals)


def safe_invoke(
    runnable: Any,
    payload: Any,
    *,
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    verbose: bool = True,
):
    """
    Safely invoke any LangChain-style runnable that has an `.invoke(...)` method.

    Works with:
    - llm.invoke(...)
    - chain.invoke(...)
    - agent_executor.invoke(...)

    Behavior:
    - Retries only on likely rate-limit / quota errors
    - Uses exponential backoff
    - Raises immediately for non-rate-limit errors
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return runnable.invoke(payload)

        except Exception as exc:
            last_error = exc

            # Non-rate-limit errors should fail fast
            if not _is_rate_limit_error(exc):
                raise

            # If this was the last attempt, stop
            if attempt == max_retries - 1:
                break

            # Exponential backoff: base_delay * 2^attempt
            delay = min(base_delay * (2**attempt), max_delay)

            # Small jitter helps avoid synchronized retries
            if jitter:
                delay += random.uniform(0, 0.75)

            if verbose:
                print(
                    f"[safe_invoke] Rate limit detected on attempt {attempt + 1}/{max_retries}. "
                    f"Sleeping {delay:.2f}s before retry..."
                )

            time.sleep(delay)

    raise RuntimeError(
        f"safe_invoke failed after {max_retries} attempts. Last error: {last_error}"
    ) from last_error

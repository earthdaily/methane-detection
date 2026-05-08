"""
Shared utilities for the methane-detection pipeline.

Centralizes retry/backoff logic used by both the STAC search step and the
per-item processing step so transient CDSE/S3/STAC failures are absorbed
inside the running container before Argo has to restart the whole pod.
"""
from __future__ import annotations

import logging
import os
import random
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Lowercased substring markers that classify an exception as a likely-transient
# remote-read error. Kept generous on purpose: a false positive only costs an
# extra retry, while a false negative surfaces as a user-visible failure.
TRANSIENT_ERROR_MARKERS: tuple[str, ...] = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "connection reset",
    "connection refused",
    "could not connect",
    "curl error",
    "expired token",
    "max session",
    "rate",
    "rate limit",
    "request timeout",
    "temporarily unavailable",
    "timeout",
    "timed out",
    "too many requests",
)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Defaults sourced from env so the workflow YAML can tune them without code
# changes. Read once at import time — same behaviour as the previous inline
# constants in process_item.py.
DEFAULT_RETRY_ATTEMPTS = _env_int("METHANE_READ_RETRY_ATTEMPTS", 4)
DEFAULT_RETRY_BASE_DELAY = _env_float("METHANE_READ_RETRY_BASE_DELAY", 5.0)
DEFAULT_RETRY_MAX_DELAY = _env_float("METHANE_READ_RETRY_MAX_DELAY", 60.0)


def is_transient_read_error(exc: BaseException) -> bool:
    """Return True for S3/STAC failures that are usually worth retrying."""
    msg = str(exc).lower()
    return any(marker in msg for marker in TRANSIENT_ERROR_MARKERS)


def retry_transient(
    operation_name: str,
    func: Callable[[], T],
    *,
    attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    log: logging.Logger | None = None,
) -> T:
    """Retry a remote-read callable with jittered exponential backoff.

    Re-raises immediately on errors that don't look transient so genuine
    failures (bad bbox, missing collection, auth) surface without delay.
    Multiple pods retrying in parallel pick different jitter offsets, which
    keeps them from hammering the upstream service at the same second.
    """
    attempts = attempts if attempts is not None else DEFAULT_RETRY_ATTEMPTS
    base_delay = base_delay if base_delay is not None else DEFAULT_RETRY_BASE_DELAY
    max_delay = max_delay if max_delay is not None else DEFAULT_RETRY_MAX_DELAY
    log = log or logger

    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts or not is_transient_read_error(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, min(delay * 0.25, 5.0))
            log.warning(
                "%s failed on attempt %s/%s with transient error: %s. Retrying in %.1fs",
                operation_name,
                attempt,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None  # loop always returns or raises before this
    raise last_exc


def write_empty_flag(path: str | None, is_empty: bool) -> None:
    """Write 'true'/'false' to a sidecar file for an Argo output parameter.

    No-op when ``path`` is falsy so the helper is safe to call unconditionally.
    Failures to write are logged but not raised — the sidecar is advisory and
    we never want to crash the producer step over a missing tmp file.
    """
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write("true" if is_empty else "false")
    except OSError as exc:
        logger.warning("Failed to write empty-flag sidecar to %s: %s", path, exc)

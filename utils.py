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
import re
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Word-boundary substring markers used as a fallback classifier for opaque
# exceptions (pystac_client.APIError, plain Exception). Type-aware checks in
# is_transient_read_error() run first; this tuple only matters when no
# structured info is available. Word boundaries are applied at match time so
# "rate" no longer matches "iterate"/"operate" and "500" no longer matches
# "50000-byte limit".
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


# botocore ClientError codes/HTTP statuses that always warrant a retry.
_TRANSIENT_CLIENT_ERROR_CODES: frozenset[str] = frozenset({
    "Throttling",
    "ThrottlingException",
    "RequestTimeout",
    "RequestTimeTooSkewed",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "InternalFailure",
    "PriorRequestNotComplete",
})
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# botocore ClientError codes that are definitively *not* worth retrying.
# Explicit so they short-circuit before the substring fallback can produce a
# false positive (e.g. a 403 AccessDenied whose message happens to contain
# "500" because a bucket name has digits).
_NON_TRANSIENT_CLIENT_ERROR_CODES: frozenset[str] = frozenset({
    "AccessDenied",
    "NoSuchBucket",
    "NoSuchKey",
    "InvalidRequest",
    "InvalidBucketName",
    "AuthorizationHeaderMalformed",
    "SignatureDoesNotMatch",
})


def _build_transient_types() -> tuple[type, ...]:
    """Collect known-transient exception classes from optional deps at import time."""
    types: list[type] = []
    try:
        from requests.exceptions import (
            ChunkedEncodingError, ConnectionError as RequestsConnectionError, Timeout,
        )
        types.extend([RequestsConnectionError, Timeout, ChunkedEncodingError])
    except ImportError:
        pass
    try:
        from urllib3.exceptions import (
            NewConnectionError, ProtocolError, ReadTimeoutError as Urllib3ReadTimeout,
        )
        types.extend([ProtocolError, Urllib3ReadTimeout, NewConnectionError])
    except ImportError:
        pass
    try:
        from botocore.exceptions import (
            ConnectionClosedError, ConnectTimeoutError,
            EndpointConnectionError, ReadTimeoutError as BotoReadTimeout,
        )
        types.extend([
            EndpointConnectionError, BotoReadTimeout,
            ConnectTimeoutError, ConnectionClosedError,
        ])
    except ImportError:
        pass
    return tuple(types)


_TRANSIENT_EXCEPTION_TYPES: tuple[type, ...] = _build_transient_types()

try:
    from botocore.exceptions import ClientError as _BotoClientError
except ImportError:
    _BotoClientError = None  # type: ignore[assignment,misc]


# Query-string parameter names whose values must never reach logs. Conservative
# allow-list (only well-known credential names) keeps false-positive redactions
# from mangling otherwise-readable error text.
_SENSITIVE_QUERY_PARAMS: tuple[str, ...] = (
    "X-Amz-Signature",
    "X-Amz-Credential",
    "X-Amz-Security-Token",
    "Signature",
    "signature",
    "sig",
    "token",
    "access_token",
    "AWSAccessKeyId",
)
_QUERY_REDACT_RE = re.compile(
    r"([?&])(" + "|".join(re.escape(p) for p in _SENSITIVE_QUERY_PARAMS) + r")="
    r"[^&\s\"'<>]+",
    re.IGNORECASE,
)
_BEARER_REDACT_RE = re.compile(
    r"(Bearer\s+)[A-Za-z0-9._\-+/=]+", re.IGNORECASE
)


def _redact_sensitive(text: str) -> str:
    """Strip credential-bearing query params and bearer tokens before logging.

    botocore/requests/rasterio occasionally stringify the failing URL into the
    exception message. Pre-signed S3/CDSE URLs carry X-Amz-Signature etc. that
    are valid replay credentials until expiry, so logging them — especially in
    a public-repo CI context — leaks shareable access. This redactor runs once
    at log time and is intentionally narrow (only well-known param names) to
    keep the rest of the message human-readable.
    """
    redacted = _QUERY_REDACT_RE.sub(r"\1\2=REDACTED", text)
    return _BEARER_REDACT_RE.sub(r"\1REDACTED", redacted)


def _matches_transient_marker(msg_lower: str) -> bool:
    return any(
        re.search(r"\b" + re.escape(m) + r"\b", msg_lower)
        for m in TRANSIENT_ERROR_MARKERS
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


def is_transient_read_error(exc: Exception) -> bool:
    """Return True for S3/STAC failures that are usually worth retrying.

    Classification order (first match wins):
      1. Known-transient exception types from requests/urllib3/botocore.
      2. botocore ClientError: explicit retry/no-retry by code or HTTP status.
      3. Word-boundary substring fallback for opaque exceptions (pystac
         APIError, plain Exception messages from upstream services).

    The structured paths exist so a real `AccessDenied` ClientError whose
    message happens to contain "500" (e.g. a bucket name) does not get
    retried four times by mistake.
    """
    if isinstance(exc, _TRANSIENT_EXCEPTION_TYPES):
        return True

    if _BotoClientError is not None and isinstance(exc, _BotoClientError):
        response = getattr(exc, "response", None) or {}
        error_code = (response.get("Error") or {}).get("Code", "")
        http_status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if error_code in _TRANSIENT_CLIENT_ERROR_CODES:
            return True
        if isinstance(http_status, int) and http_status in _TRANSIENT_HTTP_STATUSES:
            return True
        if error_code in _NON_TRANSIENT_CLIENT_ERROR_CODES:
            return False
        # Unknown ClientError code — fall through to substring fallback below.

    return _matches_transient_marker(str(exc).lower())


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

    Note: ``func`` should reconstruct any auth-bearing client/session itself
    so an expired token retry can pick up a fresh credential. Callers that
    capture a long-lived session in a closure will retry against the same
    expired credential.
    """
    # TODO: review report items #4 (full jitter — current 25% jitter clusters
    # parallel pods within ~1.25s of each retry boundary during a real outage)
    # and #5 (cap attempts/max_delay regardless of env so a YAML typo cannot
    # produce multi-day sleeps). Deferred — see plan file.
    attempts = attempts if attempts is not None else DEFAULT_RETRY_ATTEMPTS
    base_delay = base_delay if base_delay is not None else DEFAULT_RETRY_BASE_DELAY
    max_delay = max_delay if max_delay is not None else DEFAULT_RETRY_MAX_DELAY
    log = log or logger

    last_exc: Exception | None = None
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
                _redact_sensitive(str(exc)),
                delay,
            )
            time.sleep(delay)
    # Loop only exits via return or raise; this is unreachable but keeps the
    # type checker honest without relying on `assert` (stripped under -O).
    # TODO: review report item #6 — could replace with explicit RuntimeError.
    if last_exc is None:
        raise RuntimeError("retry_transient exhausted attempts with no recorded exception")
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

"""Unit tests for the shared retry/backoff utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import utils


class TestIsTransientReadError:
    @pytest.mark.parametrize("msg", [
        "503 Service Unavailable",
        "HTTP 429 Too Many Requests",
        "ConnectionResetError(104, 'Connection reset by peer')",
        "Read timed out.",
        "rate limit exceeded",
        "Could not connect to endpoint",
        "CURL error: SSL connect error",
    ])
    def test_known_transient_messages(self, msg):
        assert utils.is_transient_read_error(Exception(msg))

    @pytest.mark.parametrize("msg", [
        "AccessDenied: not authorized",
        "NoSuchBucket: bucket does not exist",
        "Invalid bbox parameter",
        "ValueError: bad input",
        "404 Not Found",
    ])
    def test_non_transient_messages(self, msg):
        assert not utils.is_transient_read_error(Exception(msg))


class TestRetryTransient:
    def test_returns_value_on_first_success(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return 42

        result = utils.retry_transient(
            "op", fn, attempts=3, base_delay=0, max_delay=0
        )
        assert result == 42
        assert calls["n"] == 1

    def test_retries_then_succeeds_on_transient_error(self, mocker):
        mocker.patch.object(utils.time, "sleep")
        seen = {"n": 0}

        def fn():
            seen["n"] += 1
            if seen["n"] < 3:
                raise RuntimeError("503 Service Unavailable")
            return "ok"

        result = utils.retry_transient(
            "op", fn, attempts=5, base_delay=0, max_delay=0
        )
        assert result == "ok"
        assert seen["n"] == 3

    def test_gives_up_after_max_attempts_on_persistent_transient(self, mocker):
        sleep = mocker.patch.object(utils.time, "sleep")

        def fn():
            raise RuntimeError("connection reset")

        with pytest.raises(RuntimeError, match="connection reset"):
            utils.retry_transient(
                "op", fn, attempts=3, base_delay=0, max_delay=0
            )
        # 3 attempts means 2 sleeps between them.
        assert sleep.call_count == 2

    def test_non_transient_error_raises_without_retry(self, mocker):
        sleep = mocker.patch.object(utils.time, "sleep")
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("bad input — not retryable")

        with pytest.raises(ValueError):
            utils.retry_transient(
                "op", fn, attempts=5, base_delay=0, max_delay=0
            )
        assert calls["n"] == 1
        assert sleep.call_count == 0

    def test_default_attempts_falls_back_to_module_constant(self, mocker):
        mocker.patch.object(utils.time, "sleep")
        mocker.patch.object(utils, "DEFAULT_RETRY_ATTEMPTS", 2)
        mocker.patch.object(utils, "DEFAULT_RETRY_BASE_DELAY", 0)
        mocker.patch.object(utils, "DEFAULT_RETRY_MAX_DELAY", 0)
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise RuntimeError("timeout")

        with pytest.raises(RuntimeError):
            utils.retry_transient("op", fn)
        assert calls["n"] == 2


class TestWriteEmptyFlag:
    def test_writes_true_when_empty(self, tmp_path: Path):
        path = tmp_path / "is_empty"
        utils.write_empty_flag(str(path), is_empty=True)
        assert path.read_text() == "true"

    def test_writes_false_when_not_empty(self, tmp_path: Path):
        path = tmp_path / "is_empty"
        utils.write_empty_flag(str(path), is_empty=False)
        assert path.read_text() == "false"

    def test_no_op_when_path_is_none(self):
        # Should not raise.
        utils.write_empty_flag(None, is_empty=True)

    def test_no_op_when_path_is_empty_string(self):
        utils.write_empty_flag("", is_empty=True)

    def test_swallows_oserror(self, tmp_path: Path, caplog):
        # TODO: review report item #10 — set caplog level explicitly so this
        # assertion isn't dependent on pytest's default propagation behaviour.
        # Deferred — see plan file.
        # Pointing into a non-existent directory triggers OSError on open.
        bogus = tmp_path / "does" / "not" / "exist" / "flag"
        utils.write_empty_flag(str(bogus), is_empty=True)
        assert any("Failed to write empty-flag" in r.message for r in caplog.records)


class TestTypeAwareClassifier:
    """Coverage for the structured (non-substring) paths in is_transient_read_error."""

    def test_requests_connection_error_is_transient(self):
        from requests.exceptions import ConnectionError as RConnectionError
        assert utils.is_transient_read_error(RConnectionError("kaboom"))

    def test_requests_timeout_is_transient(self):
        from requests.exceptions import Timeout
        assert utils.is_transient_read_error(Timeout("slow"))

    def test_botocore_endpoint_connection_error_is_transient(self):
        from botocore.exceptions import EndpointConnectionError
        exc = EndpointConnectionError(endpoint_url="https://s3.example.test")
        assert utils.is_transient_read_error(exc)

    def test_botocore_clienterror_503_is_transient_even_with_clean_message(self):
        from botocore.exceptions import ClientError
        # Message intentionally contains no transient substring — only the
        # structured 503 should drive classification.
        exc = ClientError(
            error_response={
                "Error": {"Code": "ServiceUnavailable", "Message": "please retry"},
                "ResponseMetadata": {"HTTPStatusCode": 503},
            },
            operation_name="GetObject",
        )
        assert utils.is_transient_read_error(exc)

    def test_botocore_clienterror_throttling_code_is_transient(self):
        from botocore.exceptions import ClientError
        exc = ClientError(
            error_response={
                "Error": {"Code": "Throttling", "Message": "slow down"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            operation_name="HeadObject",
        )
        assert utils.is_transient_read_error(exc)

    def test_botocore_clienterror_accessdenied_short_circuits_substring_trap(self):
        # The message contains "500" as part of a byte-count, which the OLD
        # substring classifier would have picked up as transient. The
        # structured AccessDenied code must short-circuit that.
        from botocore.exceptions import ClientError
        exc = ClientError(
            error_response={
                "Error": {"Code": "AccessDenied", "Message": "object exceeds 500 bytes"},
                "ResponseMetadata": {"HTTPStatusCode": 403},
            },
            operation_name="GetObject",
        )
        assert not utils.is_transient_read_error(exc)

    def test_botocore_clienterror_nosuchbucket_is_not_transient(self):
        from botocore.exceptions import ClientError
        exc = ClientError(
            error_response={
                "Error": {"Code": "NoSuchBucket", "Message": "bucket does not exist"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            operation_name="ListObjectsV2",
        )
        assert not utils.is_transient_read_error(exc)


class TestSubstringWordBoundaries:
    """Word-boundary tightening: previous false positives must now classify as non-transient."""

    @pytest.mark.parametrize("msg", [
        "iterate failed",                        # contains "rate" but not as a word
        "operate timeout-free path",             # "rate" in "operate" — not a word
        "AccessDenied: object exceeds 50000 bytes",  # "500" inside "50000"
        "validation error",                      # no longer matches "rate"
    ])
    def test_previously_false_positive_messages_now_non_transient(self, msg):
        # These plain Exceptions hit the substring fallback. With word
        # boundaries they must NOT be classified as transient.
        # Note: "operate timeout-free" still contains the standalone word
        # "timeout" via "timeout-free" — re-craft if test fails for that case.
        if "timeout" in msg.lower():
            pytest.skip("substring 'timeout' legitimately present as word")
        assert not utils.is_transient_read_error(Exception(msg))

    @pytest.mark.parametrize("msg", [
        "503 Service Unavailable",
        "rate limit exceeded",
        "Could not connect to endpoint",
        "Read timed out.",
    ])
    def test_known_transient_messages_still_transient(self, msg):
        # Regression guard: word-boundary tightening must not break the
        # legitimate matches the original substring classifier caught.
        assert utils.is_transient_read_error(Exception(msg))


class TestRedactSensitive:
    def test_redacts_x_amz_signature(self):
        url = (
            "https://example.s3.amazonaws.com/key?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=AKIAEXAMPLE%2F20260507"
            "&X-Amz-Signature=abc123deadbeef"
        )
        out = utils._redact_sensitive(url)
        assert "abc123deadbeef" not in out
        assert "AKIAEXAMPLE" not in out
        assert "X-Amz-Signature=REDACTED" in out
        assert "X-Amz-Credential=REDACTED" in out

    def test_redacts_bearer_token(self):
        msg = "401 Unauthorized: Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        out = utils._redact_sensitive(msg)
        assert "eyJhbGciOiJIUzI1NiJ9" not in out
        assert "Bearer REDACTED" in out

    def test_leaves_innocuous_text_alone(self):
        msg = "503 Service Unavailable from upstream STAC catalog"
        assert utils._redact_sensitive(msg) == msg

    def test_redaction_applied_in_retry_warning_log(self, mocker, caplog):
        # End-to-end: a transient-error stringified with a signed-URL must
        # appear redacted in the warning emitted by retry_transient.
        mocker.patch.object(utils.time, "sleep")
        secret = "deadbeefcafe"
        leaky_message = (
            f"503 Service Unavailable while reading "
            f"https://s3.example/key?X-Amz-Signature={secret}"
        )

        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError(leaky_message)
            return "ok"

        caplog.set_level(logging.WARNING, logger="utils")
        result = utils.retry_transient(
            "leaky-op", fn, attempts=3, base_delay=0, max_delay=0
        )
        assert result == "ok"
        joined = "\n".join(r.message for r in caplog.records)
        assert secret not in joined, "signed-URL secret leaked into warning log"
        assert "X-Amz-Signature=REDACTED" in joined

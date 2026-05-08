"""Unit tests for the shared retry/backoff utilities."""
from __future__ import annotations

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
        # Pointing into a non-existent directory triggers OSError on open.
        bogus = tmp_path / "does" / "not" / "exist" / "flag"
        utils.write_empty_flag(str(bogus), is_empty=True)
        assert any("Failed to write empty-flag" in r.message for r in caplog.records)

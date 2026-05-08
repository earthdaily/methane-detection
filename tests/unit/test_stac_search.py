"""Unit tests for stac_search.py — focused on the empty-flag and retry wiring."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pystac_client.exceptions import APIError

import stac_search


def _build_stac_client_mock(mocker, l1c_items, l2a_items):
    """Patch PyStacClient so the first .search() returns l1c_items, the second l2a_items."""
    item_collections = [
        MagicMock(items=l1c_items),
        MagicMock(items=l2a_items),
    ]
    search_results = [
        MagicMock(item_collection=MagicMock(return_value=item_collections[0])),
        MagicMock(item_collection=MagicMock(return_value=item_collections[1])),
    ]
    client = MagicMock()
    client.search.side_effect = search_results
    open_mock = mocker.patch("stac_search.PyStacClient.open", return_value=client)
    return open_mock, client


class TestEmptyFlag:
    def test_writes_true_when_l1c_search_returns_empty(self, mocker, tmp_path: Path):
        _build_stac_client_mock(mocker, l1c_items=[], l2a_items=[])
        flag = tmp_path / "is_empty"

        runner = CliRunner()
        result = runner.invoke(
            stac_search.main,
            [
                "--bbox", "[-3.67, 40.23, -3.61, 40.29]",
                "--start-datetime", "2024-01-01T00:00:00Z",
                "--end-datetime", "2024-01-31T23:59:59Z",
                "--cloud-cover", "",
                "--limit", "",
                "--catalog-url", "http://example.test/stac",
                "--empty-flag-path", str(flag),
            ],
        )
        assert result.exit_code == 0, result.output
        assert flag.read_text() == "true"
        # stdout still produces a valid JSON array so withParam doesn't choke.
        assert json.loads(result.stdout.strip().splitlines()[-1]) == []

    def test_writes_false_when_l1c_search_returns_items(
        self, mocker, tmp_path: Path, make_item, synthetic_l1c_band
    ):
        bbox = [-3.67, 40.23, -3.61, 40.29]
        l1c = make_item(
            item_id="L1C_TEST_1",
            platform="sentinel-2a",
            datetime_iso="2024-01-15T10:00:00Z",
            assets={"B11": str(synthetic_l1c_band("B11"))},
            bbox=bbox,
        )
        # deduplicate_items sorts by properties["created"]; conftest's make_item
        # doesn't set it, so add it here.
        l1c.properties["created"] = "2024-01-15T10:00:00Z"
        _build_stac_client_mock(mocker, l1c_items=[l1c], l2a_items=[])
        flag = tmp_path / "is_empty"

        runner = CliRunner()
        result = runner.invoke(
            stac_search.main,
            [
                "--bbox", json.dumps(bbox),
                "--start-datetime", "2024-01-01T00:00:00Z",
                "--end-datetime", "2024-01-31T23:59:59Z",
                "--cloud-cover", "",
                "--limit", "",
                "--catalog-url", "http://example.test/stac",
                "--empty-flag-path", str(flag),
            ],
        )
        assert result.exit_code == 0, result.output
        assert flag.read_text() == "false"

    def test_no_flag_written_when_option_omitted(self, mocker):
        _build_stac_client_mock(mocker, l1c_items=[], l2a_items=[])
        runner = CliRunner()
        result = runner.invoke(
            stac_search.main,
            [
                "--bbox", "[-3.67, 40.23, -3.61, 40.29]",
                "--start-datetime", "2024-01-01T00:00:00Z",
                "--end-datetime", "2024-01-31T23:59:59Z",
                "--cloud-cover", "",
                "--limit", "",
                "--catalog-url", "http://example.test/stac",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_terminal_api_error_exits_nonzero_and_does_not_write_flag(
        self, mocker, tmp_path: Path
    ):
        # Force every retry attempt to raise a non-transient APIError so retry
        # gives up immediately. Flag must remain unwritten so stage-2 YAML can
        # tell "API down" apart from "no scenes".
        mocker.patch(
            "stac_search.PyStacClient.open",
            side_effect=APIError("404 collection not found"),
        )
        flag = tmp_path / "is_empty"

        runner = CliRunner()
        result = runner.invoke(
            stac_search.main,
            [
                "--bbox", "[-3.67, 40.23, -3.61, 40.29]",
                "--start-datetime", "2024-01-01T00:00:00Z",
                "--end-datetime", "2024-01-31T23:59:59Z",
                "--cloud-cover", "",
                "--limit", "",
                "--catalog-url", "http://example.test/stac",
                "--empty-flag-path", str(flag),
            ],
        )
        assert result.exit_code != 0
        assert not flag.exists()


class TestRetryWiring:
    def test_search_stac_retries_on_transient_then_succeeds(self, mocker):
        # retry_transient sleeps via utils.time.sleep; stub it so the test is fast.
        import utils
        mocker.patch.object(utils.time, "sleep")

        good_collection = MagicMock(items=[])
        good_results = MagicMock(item_collection=MagicMock(return_value=good_collection))
        flaky_client = MagicMock()
        flaky_client.search.return_value = good_results

        call_count = {"n": 0}

        def open_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise APIError("503 Service Unavailable")
            return flaky_client

        mocker.patch("stac_search.PyStacClient.open", side_effect=open_side_effect)

        items = stac_search.search_stac(
            bbox=[-3.67, 40.23, -3.61, 40.29],
            start_datetime="2024-01-01T00:00:00Z",
            end_datetime="2024-01-31T23:59:59Z",
            collection="sentinel-2-l1c",
            catalog_url="http://example.test/stac",
            cloud_cover=None,
            limit=None,
            stac_provider="cdse",
        )
        assert items == []
        assert call_count["n"] == 2  # one failed + one successful retry

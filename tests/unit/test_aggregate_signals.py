"""Unit tests for aggregate_signals.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import aggregate_signals as ags


class TestComputeBoxplotStats:
    def test_empty_list_returns_none(self):
        assert ags.compute_boxplot_stats([]) is None

    def test_single_value_all_quartiles_equal(self):
        stats = ags.compute_boxplot_stats([1.234])
        assert stats == {
            "min": 1.23,
            "q1": 1.23,
            "median": 1.23,
            "q3": 1.23,
            "max": 1.23,
            "mean": 1.23,
            "count": 1,
        }

    def test_known_seven_point_list(self):
        stats = ags.compute_boxplot_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert stats["min"] == 1.0
        assert stats["max"] == 7.0
        assert stats["median"] == 4.0
        assert stats["count"] == 7
        assert stats["mean"] == 4.0
        assert stats["q1"] <= stats["median"] <= stats["q3"]


class TestExtractSignalValues:
    def test_skips_none_and_non_numeric(self):
        values = [
            {"value": 1.5},
            {"value": None},
            {"value": "not-a-number"},
            {"value": 2.0},
        ]
        assert ags.extract_signal_values(values) == [1.5, 2.0]

    def test_preserves_order(self):
        values = [{"value": v} for v in [3.0, 1.0, 2.0]]
        assert ags.extract_signal_values(values) == [3.0, 1.0, 2.0]

    def test_empty_input(self):
        assert ags.extract_signal_values([]) == []


class TestExtractDatetime:
    def test_root_datetime_wins(self):
        data = {"datetime": "2023-01-06T11:20:00Z"}
        values = [{"date": "2099-12-31T00:00:00Z"}]
        assert ags.extract_datetime(data, values) == "2023-01-06T11:20:00Z"

    def test_fallback_to_first_value_date(self):
        data = {}
        values = [{"date": "2023-01-06T11:20:00Z"}]
        assert ags.extract_datetime(data, values) == "2023-01-06T11:20:00Z"

    def test_none_when_no_datetime_anywhere(self):
        assert ags.extract_datetime({}, []) is None


class TestRecordFromData:
    def test_new_payload_shape(self):
        """Current producer writes ``{"datetime": ..., "values": [...]}``."""
        data = {
            "datetime": "2023-01-06T11:20:00Z",
            "values": [{"value": 0.1}, {"value": 0.2}, {"value": 0.3}],
        }
        record = ags.record_from_data(data)
        assert record is not None
        assert record["datetime"] == "2023-01-06T11:20:00Z"
        assert record["count"] == 3

    def test_legacy_payload_shape(self):
        """Old producer wrote datetime on each value item; aggregator must still cope."""
        data = {
            "values": [
                {"date": "2023-01-06T11:20:00Z", "value": 0.1},
                {"date": "2023-01-06T11:20:00Z", "value": 0.2},
            ]
        }
        record = ags.record_from_data(data)
        assert record is not None
        assert record["datetime"] == "2023-01-06T11:20:00Z"

    def test_missing_datetime_returns_none(self):
        data = {"values": [{"value": 0.1}]}
        assert ags.record_from_data(data) is None

    def test_no_valid_signals_returns_none(self):
        data = {"datetime": "2023-01-06T11:20:00Z", "values": []}
        assert ags.record_from_data(data) is None


class TestIterSignalFiles:
    def test_deterministic_sort(self, tmp_path: Path):
        for name in ["c_time_signal.json", "a_time_signal.json", "b_time_signal.json"]:
            (tmp_path / name).write_text("{}")
        files = ags.iter_signal_files(tmp_path)
        assert [p.name for p in files] == sorted([p.name for p in files])

    def test_ignores_non_signal_files(self, tmp_path: Path):
        (tmp_path / "a_time_signal.json").write_text("{}")
        (tmp_path / "unrelated.json").write_text("{}")
        (tmp_path / "b.txt").write_text("x")
        files = ags.iter_signal_files(tmp_path)
        assert len(files) == 1
        assert files[0].name == "a_time_signal.json"


class TestWriteAggregatedOutput:
    def test_creates_dir_and_writes_valid_json(self, tmp_path: Path):
        signals_dir = tmp_path / "signals"
        records = [{"datetime": "2023-01-06T11:20:00Z", "mean": 0.1}]
        out_file = ags.write_aggregated_output(signals_dir, records)
        assert out_file.exists()
        payload = json.loads(out_file.read_text())
        assert payload == {"data": {"values": records}}


class TestMainCli:
    def test_empty_assets_dir_produces_empty_output(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        signals_dir = tmp_path / "signals"
        assets_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            ags.main,
            ["--assets-dir", str(assets_dir), "--signals-dir", str(signals_dir)],
        )
        assert result.exit_code == 0
        out = json.loads((signals_dir / "items_time_signal.json").read_text())
        assert out == {"data": {"values": []}}

    def test_multiple_items_are_sorted_by_datetime(self, tmp_path: Path):
        assets_dir = tmp_path / "assets"
        signals_dir = tmp_path / "signals"
        assets_dir.mkdir()

        (assets_dir / "later_time_signal.json").write_text(
            json.dumps(
                {
                    "datetime": "2023-02-01T00:00:00Z",
                    "values": [{"value": 0.2}],
                }
            )
        )
        (assets_dir / "earlier_time_signal.json").write_text(
            json.dumps(
                {
                    "datetime": "2023-01-01T00:00:00Z",
                    "values": [{"value": 0.1}],
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            ags.main,
            ["--assets-dir", str(assets_dir), "--signals-dir", str(signals_dir)],
        )
        assert result.exit_code == 0
        out = json.loads((signals_dir / "items_time_signal.json").read_text())
        datetimes = [r["datetime"] for r in out["data"]["values"]]
        assert datetimes == sorted(datetimes)
        assert len(datetimes) == 2

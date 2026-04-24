"""Unit tests for the full pipeline orchestration CLI."""
from __future__ import annotations

from click.testing import CliRunner

import run_pipeline


def test_pipeline_wires_search_process_catalog_and_aggregation(monkeypatch, mocker):
    l1c_item = mocker.Mock(id="S2A_L1C_TEST")
    l2a_item = mocker.Mock(id="S2A_L2A_TEST")
    search_calls = []

    def fake_search_stac(**kwargs):
        search_calls.append(kwargs)
        if kwargs["collection"] == "sentinel-2-l1c":
            return [l1c_item]
        return [l2a_item]

    monkeypatch.setattr(run_pipeline.stac_search, "search_stac", fake_search_stac)
    monkeypatch.setattr(run_pipeline.stac_search, "deduplicate_items", lambda items: items)
    monkeypatch.setattr(
        run_pipeline.stac_search,
        "build_output_payload",
        lambda _l1c, _l2a: [
            {"sentinel-2-l1c": "S2A_L1C_TEST", "sentinel-2-l2a": "S2A_L2A_TEST"}
        ],
    )
    process_mock = mocker.Mock()
    catalog_mock = mocker.Mock()
    aggregate_mock = mocker.Mock()
    monkeypatch.setattr(run_pipeline, "run_process_item", process_mock)
    monkeypatch.setattr(run_pipeline.process_item, "write_stac_catalog", catalog_mock)
    monkeypatch.setattr(run_pipeline.aggregate_signals, "main", aggregate_mock)

    result = CliRunner().invoke(
        run_pipeline.main,
        [
            "--bbox",
            "[-3.67, 40.23, -3.61, 40.29]",
            "--catalog_url",
            "https://stac.example.com",
            "--limit",
            "1",
            "--cloud_cover",
            "5",
            "--skip-viz",
            "--skip-colorized",
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call["collection"] for call in search_calls] == [
        "sentinel-2-l1c",
        "sentinel-2-l2a",
    ]
    process_mock.assert_called_once_with(
        bbox=[-3.67, 40.23, -3.61, 40.29],
        collection="sentinel-2-l1c",
        l1c_id="S2A_L1C_TEST",
        l2a_id="S2A_L2A_TEST",
        download_bands_list=["B11.jp2", "B12.jp2"],
        skip_viz=True,
        skip_colorized=True,
        skip_overviews=False,
    )
    catalog_mock.assert_called_once_with(["S2A_L1C_TEST"])
    aggregate_mock.assert_called_once_with(
        args=["--assets-dir", "out/assets", "--signals-dir", "out/signals"],
        standalone_mode=False,
    )


def test_pipeline_requires_catalog_url(monkeypatch):
    monkeypatch.delenv("CATALOG_URL", raising=False)

    result = CliRunner().invoke(
        run_pipeline.main,
        ["--bbox", "[-3.67, 40.23, -3.61, 40.29]"],
    )

    assert result.exit_code == 1

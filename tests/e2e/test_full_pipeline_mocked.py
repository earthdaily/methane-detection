"""
Offline end-to-end test: drive the whole ``process_item.main`` pipeline
with synthetic rasters + mocked STAC + mocked AWSSession. Then run
``aggregate_signals.main`` over the collected time-signal JSONs.

No network. No AWS. Deterministic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from click.testing import CliRunner

import aggregate_signals as ags
import process_item as pi


pytestmark = pytest.mark.e2e_mocked


def _redirect_outputs(monkeypatch, out_dir: Path) -> None:
    """Point the module-level output dirs at a tmp location."""
    stac_items = out_dir / "stac_items"
    assets = out_dir / "assets"
    monkeypatch.setattr(pi, "OUT_DIR", str(out_dir))
    monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(stac_items))
    monkeypatch.setattr(pi, "ASSETS_OUT", str(assets))


@pytest.fixture
def mocked_pipeline(
    tmp_path: Path,
    monkeypatch,
    synthetic_l1c_band,
    synthetic_l2a_scl: Path,
    synthetic_l2a_visual: Path,
    make_item,
    patch_stac_client,
    patch_aws_session,
):
    """
    Wire up a self-contained run: synthetic L1C + L2A rasters, a fake
    STAC catalog, neutralized AWS session, and output dirs redirected
    to tmp_path.
    """
    out_dir = tmp_path / "out"
    _redirect_outputs(monkeypatch, out_dir)
    monkeypatch.setenv("CATALOG_URL", "https://stac.example.com")

    b11 = synthetic_l1c_band("B11", mean=2500.0, seed=1)
    b12 = synthetic_l1c_band("B12", mean=2300.0, seed=2)

    l1c_id = "S2A_L1C_MOCK_001"
    l2a_id = "S2A_L2A_MOCK_001"
    # Positioned so bbox + its doubled expansion both fit inside the
    # synthetic raster WGS84 footprint defined in conftest.
    bbox = [-3.83, 40.32, -3.77, 40.38]

    l1c_item = make_item(
        item_id=l1c_id,
        platform="sentinel-2a",
        datetime_iso="2023-01-06T11:20:00Z",
        assets={"B11.jp2": str(b11), "B12.jp2": str(b12)},
        bbox=bbox,
    )
    l2a_item = make_item(
        item_id=l2a_id,
        platform="sentinel-2a",
        datetime_iso="2023-01-06T11:20:00Z",
        assets={"scl": str(synthetic_l2a_scl), "visual": str(synthetic_l2a_visual)},
        bbox=bbox,
    )
    patch_stac_client([l1c_item, l2a_item])

    return {
        "out_dir": out_dir,
        "assets_dir": out_dir / "assets",
        "stac_items_dir": out_dir / "stac_items",
        "l1c_id": l1c_id,
        "l2a_id": l2a_id,
        "bbox": bbox,
    }


def test_full_pipeline_l1c_only(mocked_pipeline):
    """Smoke: run main with just an L1C item, no cloud masking."""
    info = mocked_pipeline
    runner = CliRunner()
    result = runner.invoke(
        pi.main,
        [
            "--bbox",
            json.dumps(info["bbox"]),
            "--collection",
            "sentinel-2-l1c",
            "--l1c-id",
            info["l1c_id"],
            "--skip-viz",
            "--skip-colorized",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    assets = info["assets_dir"]
    methane_tif = assets / f"{info['l1c_id']}_methane_enhancement.tif"
    averaged_tif = assets / f"{info['l1c_id']}_averaged_methane_enhancement.tif"
    time_signal = assets / f"{info['l1c_id']}_time_signal.json"
    stac_item = info["stac_items_dir"] / f"{info['l1c_id']}.json"

    assert methane_tif.exists()
    assert averaged_tif.exists()
    assert time_signal.exists()
    assert stac_item.exists()

    # Methane TIFF contract: EPSG:4326, a floating-point dtype, no overviews.
    # The exact float width (32 vs 64) depends on dtype propagation from the
    # spectral template; the policy is only that dtype is preserved end-to-end.
    with rasterio.open(methane_tif) as ds:
        assert ds.crs.to_epsg() == 4326
        assert ds.dtypes[0] in ("float32", "float64")
        assert ds.overviews(1) == []

    # Time signal JSON uses the new root-level ``datetime`` field.
    payload = json.loads(time_signal.read_text())
    assert payload["datetime"] == "2023-01-06T11:20:00Z"
    assert isinstance(payload["values"], list)


def test_full_pipeline_with_l2a_cloud_mask(mocked_pipeline):
    """Run with an L2A id; expect an RGB visual TIFF alongside methane outputs."""
    info = mocked_pipeline
    runner = CliRunner()
    result = runner.invoke(
        pi.main,
        [
            "--bbox",
            json.dumps(info["bbox"]),
            "--collection",
            "sentinel-2-l1c",
            "--l1c-id",
            info["l1c_id"],
            "--l2a-id",
            info["l2a_id"],
            "--skip-viz",
            "--skip-colorized",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    rgb_path = info["assets_dir"] / f"{info['l2a_id']}_rgb_visual.tif"
    assert rgb_path.exists()
    # Regression guard: old create_ortho forced float32 on uint8 RGB input.
    with rasterio.open(rgb_path) as ds:
        assert ds.count == 3
        assert ds.dtypes == ("uint8", "uint8", "uint8")
        assert ds.overviews(1) == []


def test_aggregate_across_multiple_items(
    tmp_path: Path,
    monkeypatch,
    synthetic_l1c_band,
    make_item,
    patch_stac_client,
    patch_aws_session,
):
    """
    Run ``process_item`` twice (two different dates), then aggregate.
    The aggregate JSON must have one record per item sorted by datetime.
    """
    out_dir = tmp_path / "out"
    _redirect_outputs(monkeypatch, out_dir)
    monkeypatch.setenv("CATALOG_URL", "https://stac.example.com")

    # Positioned so bbox + its doubled expansion both fit inside the
    # synthetic raster WGS84 footprint defined in conftest.
    bbox = [-3.83, 40.32, -3.77, 40.38]
    runs = [
        ("S2A_L1C_ITEM_A", "2023-01-06T11:20:00Z", 1),
        ("S2A_L1C_ITEM_B", "2023-01-21T11:19:58Z", 2),
    ]

    for item_id, dt_iso, seed in runs:
        b11 = synthetic_l1c_band(f"B11_{seed}", mean=2500.0, seed=seed)
        b12 = synthetic_l1c_band(f"B12_{seed}", mean=2300.0, seed=seed + 10)
        item = make_item(
            item_id=item_id,
            platform="sentinel-2a",
            datetime_iso=dt_iso,
            assets={"B11.jp2": str(b11), "B12.jp2": str(b12)},
            bbox=bbox,
        )
        patch_stac_client([item])

        result = CliRunner().invoke(
            pi.main,
            [
                "--bbox",
                json.dumps(bbox),
                "--collection",
                "sentinel-2-l1c",
                "--l1c-id",
                item_id,
                "--skip-viz",
                "--skip-colorized",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    # Aggregate
    signals_dir = tmp_path / "signals"
    ags_result = CliRunner().invoke(
        ags.main,
        [
            "--assets-dir",
            str(out_dir / "assets"),
            "--signals-dir",
            str(signals_dir),
        ],
    )
    assert ags_result.exit_code == 0, ags_result.output

    out = json.loads((signals_dir / "items_time_signal.json").read_text())
    records = out["data"]["values"]
    assert len(records) == 2
    datetimes = [r["datetime"] for r in records]
    assert datetimes == sorted(datetimes)
    # All stats are finite numbers.
    for rec in records:
        for key in ("min", "q1", "median", "q3", "max", "mean"):
            assert isinstance(rec[key], (int, float))
            assert np.isfinite(rec[key])

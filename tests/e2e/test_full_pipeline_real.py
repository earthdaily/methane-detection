"""
Live end-to-end test: exercises the full Argo-style pipeline against a real
STAC catalog and requester-pays S3.

    stac_search.py  ->  process_item.py (per id)  ->  aggregate_signals.py

Skipped by default. To execute, set the same env vars ``process_item.py``
itself reads at runtime:

    export CATALOG_URL="https://earth-search.aws.element84.com/v1"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_REGION="us-west-2"   # Sentinel-2 L1C/L2A bucket region

Run with: ``pytest -m e2e_real``

Test parameters (bbox / date range / cloud cover / limit) are hard-coded
below to the values used for routine QA of the methane-detection pipeline.
These are known to return a workable set of Sentinel-2 items year after
year.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import rasterio


pytestmark = pytest.mark.e2e_real


# Pipeline runtime env vars that match process_item.py's expectations.
REQUIRED_PIPELINE_VARS = (
    "CATALOG_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

# Test parameters — the well-tested defaults used for ongoing QA.
# Overridable via env vars if you want to run the same harness against a
# different AOI / time range / filter.
DEFAULT_BBOX = "[-3.67, 40.23, -3.61, 40.29]"
DEFAULT_START = "2023-01-01T00:00:00Z"
DEFAULT_END = "2023-01-31T23:59:59Z"
DEFAULT_CLOUD_COVER = "10"
DEFAULT_LIMIT = "10"


REPO_ROOT = Path(__file__).resolve().parents[2]
STAC_SEARCH = REPO_ROOT / "stac_search.py"
PROCESS_ITEM = REPO_ROOT / "process_item.py"
AGGREGATE_SIGNALS = REPO_ROOT / "aggregate_signals.py"


def _check_env_or_skip() -> None:
    missing = [v for v in REQUIRED_PIPELINE_VARS if not os.getenv(v)]
    if missing:
        pytest.skip(
            "Real E2E skipped. Missing required env vars: "
            + ", ".join(missing)
            + ". See tests/e2e/README.md for setup."
        )


def _params() -> dict[str, str]:
    return {
        "bbox": os.environ.get("METHANE_E2E_BBOX", DEFAULT_BBOX),
        "start": os.environ.get("METHANE_E2E_START", DEFAULT_START),
        "end": os.environ.get("METHANE_E2E_END", DEFAULT_END),
        "cloud_cover": os.environ.get("METHANE_E2E_CLOUD_COVER", DEFAULT_CLOUD_COVER),
        "limit": os.environ.get("METHANE_E2E_LIMIT", DEFAULT_LIMIT),
    }


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _stac_search(params: dict[str, str]) -> list[dict[str, str | None]]:
    """Run stac_search.py and parse its stdout JSON payload."""
    result = _run(
        [
            sys.executable,
            str(STAC_SEARCH),
            "--bbox", params["bbox"],
            "--start_datetime", params["start"],
            "--end_datetime", params["end"],
            "--cloud_cover", params["cloud_cover"],
            "--limit", params["limit"],
        ]
    )
    assert result.returncode == 0, (
        f"stac_search.py failed (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    payload = json.loads(result.stdout.strip())
    assert isinstance(payload, list), f"stac_search output is not a list: {payload!r}"
    return payload


def _run_process_item(
    tmp_path: Path,
    bbox: str,
    l1c_id: str,
    l2a_id: str | None,
) -> None:
    args = [
        sys.executable, str(PROCESS_ITEM),
        "--bbox", bbox,
        "--collection", "sentinel-2-l1c",
        "--l1c-id", l1c_id,
        "--skip-viz",
        "--skip-colorized",
    ]
    if l2a_id:
        args += ["--l2a-id", l2a_id]
    result = _run(args, cwd=tmp_path, timeout=900)
    assert result.returncode == 0, (
        f"process_item.py failed for {l1c_id} (exit {result.returncode}).\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_stac_search_returns_items():
    """Thin smoke that STAC search yields at least one paired item.

    Kept separate from the heavy processing test so a catalog-side issue
    (missing scenes, 5xx, auth) fails fast and visibly.
    """
    _check_env_or_skip()
    payload = _stac_search(_params())
    assert len(payload) >= 1, "STAC search returned no items for the default AOI/window"
    for entry in payload:
        assert "sentinel-2-l1c" in entry
        assert entry["sentinel-2-l1c"], f"empty L1C id in entry: {entry}"


def test_full_pipeline_search_process_aggregate(tmp_path: Path):
    """
    Full chain: stac_search.py -> process_item.py (per id) -> aggregate_signals.py.

    Matches the Argo workflow topology. Asserts all per-item outputs exist
    and the aggregator emits one record per processed item.
    """
    _check_env_or_skip()
    params = _params()

    payload = _stac_search(params)
    assert len(payload) >= 1, "No items to process"

    processed_ids: list[str] = []
    for entry in payload:
        l1c_id = entry["sentinel-2-l1c"]
        l2a_id = entry.get("sentinel-2-l2a")
        _run_process_item(tmp_path, params["bbox"], l1c_id, l2a_id)
        processed_ids.append(l1c_id)

    assets_dir = tmp_path / "out" / "assets"
    stac_items_dir = tmp_path / "out" / "stac_items"

    for l1c_id in processed_ids:
        methane_tif = assets_dir / f"{l1c_id}_methane_enhancement.tif"
        averaged_tif = assets_dir / f"{l1c_id}_averaged_methane_enhancement.tif"
        time_signal = assets_dir / f"{l1c_id}_time_signal.json"
        item_meta = stac_items_dir / f"{l1c_id}.json"

        assert methane_tif.exists(), f"missing: {methane_tif}"
        assert averaged_tif.exists(), f"missing: {averaged_tif}"
        assert time_signal.exists(), f"missing: {time_signal}"
        assert item_meta.exists(), f"missing: {item_meta}"

        with rasterio.open(methane_tif) as ds:
            assert ds.crs.to_epsg() == 4326
            assert ds.count == 1
            assert ds.dtypes[0] in ("float32", "float64")
            assert ds.overviews(1) == []

        sig_payload = json.loads(time_signal.read_text())
        assert sig_payload.get("datetime"), "time-signal JSON missing root-level datetime"
        assert isinstance(sig_payload.get("values"), list)

    # Aggregate and verify one record per processed item.
    signals_dir = tmp_path / "signals"
    agg = _run(
        [
            sys.executable, str(AGGREGATE_SIGNALS),
            "--assets-dir", str(assets_dir),
            "--signals-dir", str(signals_dir),
        ],
        timeout=60,
    )
    assert agg.returncode == 0, (
        f"aggregate_signals.py failed (exit {agg.returncode}).\n"
        f"STDOUT:\n{agg.stdout}\n"
        f"STDERR:\n{agg.stderr}"
    )

    out = json.loads((signals_dir / "items_time_signal.json").read_text())
    records = out["data"]["values"]
    assert len(records) == len(processed_ids)
    for rec in records:
        assert "datetime" in rec
        for k in ("min", "q1", "median", "q3", "max", "mean", "count"):
            assert k in rec

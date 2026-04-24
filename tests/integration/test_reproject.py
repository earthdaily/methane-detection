"""
Integration tests for ``read_and_reproject_data``.

Exercises the full reprojection + STAC item resolution path against
synthetic rasters on disk, with PyStacClient mocked out. No network,
no AWS — just GDAL/rasterio.
"""
from __future__ import annotations

from pathlib import Path

import affine
import numpy as np
import pytest
import rasterio

import process_item as pi


pytestmark = pytest.mark.integration


def _make_l1c_item(make_item, l1c_dir: Path, band_paths: dict[str, Path]):
    return make_item(
        item_id="S2A_L1C_TEST",
        platform="sentinel-2a",
        datetime_iso="2023-01-06T11:20:00Z",
        assets={name: str(path) for name, path in band_paths.items()},
        bbox=[-3.90, 40.25, -3.65, 40.45],
    )


def test_reads_l1c_and_returns_expected_shape(
    tmp_path: Path,
    synthetic_l1c_band,
    make_item,
    patch_stac_client,
    monkeypatch,
):
    monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(tmp_path / "stac_items"))
    (tmp_path / "stac_items").mkdir()

    b11 = synthetic_l1c_band("B11", mean=2500.0, seed=1)
    b12 = synthetic_l1c_band("B12", mean=2300.0, seed=2)
    item = _make_l1c_item(make_item, tmp_path, {"B11.jp2": b11, "B12.jp2": b12})
    patch_stac_client([item])

    bbox = [-3.83, 40.32, -3.77, 40.38]
    result = pi.read_and_reproject_data(
        bbox=bbox,
        collection="sentinel-2-l1c",
        item_id="S2A_L1C_TEST",
        catalog_url="https://stac.example.com",
        download_bands_list=["B11.jp2", "B12.jp2"],
        aws_session=None,
    )
    assert result is not None
    out_img, platform, dst_tfm, width, height, item_dict = result

    assert platform == "sentinel-2a"
    assert out_img.shape == (2, height, width)
    assert out_img.dtype == np.float32
    assert isinstance(dst_tfm, affine.Affine)
    # Output CRS is EPSG:4326 regardless of input CRS; transform pixel size is in degrees.
    assert abs(dst_tfm[0]) < 1.0
    # Item metadata is returned for the final output STAC writer.
    assert item_dict["id"] == "S2A_L1C_TEST"
    assert item_dict["properties"]["platform"] == "sentinel-2a"


def test_transform_properties_overrides_derivation(
    tmp_path: Path,
    synthetic_l1c_band,
    make_item,
    patch_stac_client,
    monkeypatch,
):
    """
    When ``transform_properties`` is passed (the L2A path), the function MUST
    reuse the given transform/width/height instead of calling
    ``calculate_default_transform``. This is the guarantee that L1C and L2A
    end up on the same grid within a single run.
    """
    monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(tmp_path / "stac_items"))
    (tmp_path / "stac_items").mkdir()

    b11 = synthetic_l1c_band("B11", seed=1)
    b12 = synthetic_l1c_band("B12", seed=2)
    item = _make_l1c_item(make_item, tmp_path, {"B11.jp2": b11, "B12.jp2": b12})
    patch_stac_client([item])

    fixed_tfm = affine.Affine(0.0001, 0.0, -3.83, 0.0, -0.0001, 40.38)
    fixed_width = 1200
    fixed_height = 1200

    result = pi.read_and_reproject_data(
        bbox=[-3.83, 40.32, -3.77, 40.38],
        collection="sentinel-2-l1c",
        item_id="S2A_L1C_TEST",
        catalog_url="https://stac.example.com",
        download_bands_list=["B11.jp2", "B12.jp2"],
        aws_session=None,
        transform_properties=(fixed_tfm, fixed_width, fixed_height),
    )
    assert result is not None
    out_img, _platform, dst_tfm, width, height, _item = result
    assert dst_tfm == fixed_tfm
    assert width == fixed_width
    assert height == fixed_height
    assert out_img.shape == (2, fixed_height, fixed_width)


def test_visual_multiband_asset_expands_output_channels(
    tmp_path: Path,
    synthetic_l1c_band,
    synthetic_l2a_scl: Path,
    synthetic_l2a_visual: Path,
    make_item,
    patch_stac_client,
    monkeypatch,
):
    """
    The ``visual`` asset is actually a 3-band RGB. The code path in
    ``read_and_reproject_data`` expands ``num_bands`` by 2 when the string
    ``"visual"`` appears in ``download_bands_list``. Verify the output array
    shape matches.
    """
    monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(tmp_path / "stac_items"))
    (tmp_path / "stac_items").mkdir()

    item = make_item(
        item_id="S2A_L2A_TEST",
        platform="sentinel-2a",
        datetime_iso="2023-01-06T11:20:00Z",
        assets={
            "scl": str(synthetic_l2a_scl),
            "visual": str(synthetic_l2a_visual),
        },
        bbox=[-3.90, 40.25, -3.65, 40.45],
    )
    patch_stac_client([item])

    # Reuse a fixed transform so we don't depend on calculate_default_transform.
    fixed_tfm = affine.Affine(0.0001, 0.0, -3.83, 0.0, -0.0001, 40.38)
    width, height = 600, 600

    result = pi.read_and_reproject_data(
        bbox=[-3.83, 40.32, -3.77, 40.38],
        collection="sentinel-2-l2a",
        item_id="S2A_L2A_TEST",
        catalog_url="https://stac.example.com",
        download_bands_list=["scl", "visual"],
        aws_session=None,
        transform_properties=(fixed_tfm, width, height),
    )
    assert result is not None
    out_img, _platform, _tfm, _w, _h, _item = result
    # 2 download entries (scl + visual), but visual is itself 3 bands => 4 total channels.
    assert out_img.shape == (4, height, width)


def test_missing_platform_returns_none(
    tmp_path: Path,
    synthetic_l1c_band,
    patch_stac_client,
    monkeypatch,
):
    """
    If a STAC item has no ``platform`` property, ``read_and_reproject_data``
    must return None instead of raising. That was explicit error-handling in
    the source.
    """
    import pystac
    from datetime import datetime, timezone

    monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(tmp_path / "stac_items"))
    (tmp_path / "stac_items").mkdir()

    b11 = synthetic_l1c_band("B11")
    item = pystac.Item(
        id="NO_PLATFORM",
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=[0, 0, 1, 1],
        datetime=datetime(2023, 1, 1, tzinfo=timezone.utc),
        properties={"datetime": "2023-01-01T00:00:00Z"},  # no platform key
    )
    item.add_asset("B11.jp2", pystac.Asset(href=str(b11)))
    patch_stac_client([item])

    result = pi.read_and_reproject_data(
        bbox=[0, 0, 1, 1],
        collection="sentinel-2-l1c",
        item_id="NO_PLATFORM",
        catalog_url="https://stac.example.com",
        download_bands_list=["B11.jp2"],
        aws_session=None,
    )
    assert result is None

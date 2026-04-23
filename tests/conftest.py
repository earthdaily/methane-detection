"""
Shared fixtures for the methane-detection test suite.

Provides synthetic Sentinel-2 GeoTIFFs and a helper to build fake STAC items
that point at those files via ``file://`` URLs, so tests can exercise the
real reprojection path without any network or S3 calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import affine
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.crs import CRS

# A UTM 30N window roughly over central Spain. Sized so the full synthetic
# footprint (~25km square) comfortably covers the test bbox + its doubled
# expansion, which is what ``process_regional_signal`` needs to see data
# both inside and outside the AOI.
SYNTHETIC_UTM_CRS = CRS.from_epsg(32630)
SYNTHETIC_WIDTH = 128
SYNTHETIC_HEIGHT = 128
SYNTHETIC_PIXEL_SIZE = 200.0  # meters per pixel -> 25.6 km raster footprint
SYNTHETIC_ORIGIN_X = 420000.0
SYNTHETIC_ORIGIN_Y = 4480000.0


def _utm_transform() -> affine.Affine:
    """Affine transform for all synthetic rasters (shared so L1C and L2A align)."""
    return affine.Affine(
        SYNTHETIC_PIXEL_SIZE,
        0.0,
        SYNTHETIC_ORIGIN_X,
        0.0,
        -SYNTHETIC_PIXEL_SIZE,
        SYNTHETIC_ORIGIN_Y,
    )


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    dtype: str,
    nodata: float | int | None = 0,
) -> Path:
    """Write a plain (non-COG) GeoTIFF that rasterio.open can read back."""
    if data.ndim == 2:
        count = 1
        height, width = data.shape
        write_data = data[np.newaxis, ...]
    else:
        count, height, width = data.shape
        write_data = data

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "crs": SYNTHETIC_UTM_CRS,
        "transform": _utm_transform(),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(write_data.astype(dtype))
    return path


@pytest.fixture
def synthetic_l1c_band(tmp_path: Path) -> Callable[[str, float], Path]:
    """
    Factory that writes a single-band uint16 L1C-style GeoTIFF to tmp_path.

    Usage: ``path = synthetic_l1c_band("B11", mean=2500.0)``.
    """

    def _make(name: str, mean: float = 2500.0, seed: int = 0) -> Path:
        rng = np.random.default_rng(seed)
        data = rng.normal(loc=mean, scale=200.0, size=(SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH))
        data = np.clip(data, 1, 10000).astype(np.uint16)
        return _write_geotiff(tmp_path / f"{name}.tif", data, dtype="uint16")

    return _make


@pytest.fixture
def synthetic_l2a_scl(tmp_path: Path) -> Path:
    """Scene classification raster covering a mix of clear and cloudy codes."""
    scl = np.full((SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH), fill_value=4, dtype=np.uint8)
    scl[:32, :] = 9  # cloud high probability (not in CLEAR_MASK_CODES)
    scl[32:64, :32] = 5  # not-vegetated (clear)
    scl[64:, 64:] = 11  # snow (clear)
    return _write_geotiff(tmp_path / "SCL.tif", scl, dtype="uint8", nodata=0)


@pytest.fixture
def synthetic_l2a_visual(tmp_path: Path) -> Path:
    """Three-band uint8 RGB visual raster."""
    rng = np.random.default_rng(42)
    visual = rng.integers(50, 230, size=(3, SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH), dtype=np.uint8)
    return _write_geotiff(tmp_path / "TCI.tif", visual, dtype="uint8", nodata=0)


def _make_item(
    item_id: str,
    platform: str,
    datetime_iso: str,
    assets: dict[str, str],
    bbox: list[float],
) -> pystac.Item:
    """Build a minimal pystac.Item with file:// asset hrefs."""
    geom = {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }
    item = pystac.Item(
        id=item_id,
        geometry=geom,
        bbox=bbox,
        datetime=datetime.fromisoformat(datetime_iso.replace("Z", "+00:00")),
        properties={"platform": platform, "datetime": datetime_iso},
    )
    for key, href in assets.items():
        item.add_asset(key, pystac.Asset(href=href, media_type="image/tiff"))
    return item


@pytest.fixture
def make_item() -> Callable[..., pystac.Item]:
    """Factory fixture exposing ``_make_item`` for tests to parameterize."""
    return _make_item


@pytest.fixture
def patch_stac_client(mocker):
    """
    Patch ``process_item.PyStacClient`` so ``client.search(ids=[...]).item_collection().items``
    returns only the caller-registered items whose ids match the search query.

    Each call replaces the previous registration.

    Returns a setter callable: ``patch_stac_client([item1, item2])``.
    """
    registered: dict[str, pystac.Item] = {}

    def _install(items: list[pystac.Item]) -> Any:
        registered.clear()
        for it in items:
            registered[it.id] = it

        def _search(*args, **kwargs):
            wanted_ids = kwargs.get("ids") or []
            matched = [registered[i] for i in wanted_ids if i in registered]
            search_result = mocker.MagicMock()
            search_result.item_collection.return_value.items = matched
            return search_result

        mock_client = mocker.patch("process_item.PyStacClient")
        mock_client.open.return_value.search.side_effect = _search
        return mock_client

    return _install


@pytest.fixture
def patch_aws_session(mocker):
    """Neutralize AWSSession + boto3 so no real AWS calls are attempted."""
    mocker.patch("process_item.boto3.Session", return_value=mocker.MagicMock())
    mocker.patch("process_item.AWSSession", return_value=None)

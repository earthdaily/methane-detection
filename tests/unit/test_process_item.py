"""Unit tests for pure functions and I/O helpers in process_item.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import click
import numpy as np
import pytest
import rasterio

import process_item as pi


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

class TestDoubleBbox:
    def test_symmetric_bbox_triples_dimensions(self):
        result = pi.double_bbox([0.0, 0.0, 2.0, 2.0])
        assert result == pytest.approx([-1.0, -1.0, 3.0, 3.0])
        # center preserved at (1, 1), width/height 3x the original half-extent
        assert (result[0] + result[2]) / 2 == pytest.approx(1.0)
        assert (result[1] + result[3]) / 2 == pytest.approx(1.0)

    def test_asymmetric_bbox_preserves_center_and_doubles_extent(self):
        bbox = [-3.67, 40.23, -3.61, 40.29]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        result = pi.double_bbox(bbox)
        new_center = [(result[0] + result[2]) / 2, (result[1] + result[3]) / 2]
        assert new_center == pytest.approx(center)
        # Each side extends outward by half the original extent => total width 2x.
        assert (result[2] - result[0]) == pytest.approx(2 * (bbox[2] - bbox[0]))
        assert (result[3] - result[1]) == pytest.approx(2 * (bbox[3] - bbox[1]))


class TestBuildCloudMask:
    @pytest.mark.parametrize("clear_code", [4, 5, 6, 7, 11])
    def test_each_clear_code_is_unmasked(self, clear_code):
        scl = np.full((8, 8), fill_value=clear_code, dtype=np.uint8)
        mask = pi.build_cloud_mask(scl)
        assert mask.dtype == scl.dtype
        assert np.all(mask == 1)

    @pytest.mark.parametrize("cloudy_code", [0, 1, 2, 3, 8, 9, 10])
    def test_cloudy_codes_are_masked(self, cloudy_code):
        scl = np.full((8, 8), fill_value=cloudy_code, dtype=np.uint8)
        mask = pi.build_cloud_mask(scl)
        assert np.all(mask == 0)

    def test_mixed_scene(self):
        scl = np.array([[4, 9], [5, 8]], dtype=np.uint8)
        mask = pi.build_cloud_mask(scl)
        assert mask.tolist() == [[1, 0], [1, 0]]


class TestCreateImeMask:
    def test_top_five_percent_are_masked(self):
        mf = np.arange(100, dtype=np.float32).reshape(10, 10)
        mask = pi.create_ime_mask(mf)
        # 95th percentile of 0..99 is 94.05; pixels strictly greater are masked => 95..99 = 5 pixels
        assert mask.sum() == 5
        assert np.all(mask[mf > np.percentile(mf, 95)] == 1)

    def test_all_zero_input(self):
        mf = np.zeros((5, 5), dtype=np.float32)
        mask = pi.create_ime_mask(mf)
        # No pixel is strictly greater than the 95th percentile (which is also 0).
        assert mask.sum() == 0


class TestParseListString:
    def test_valid_json_list(self):
        assert pi.parse_list_string(None, None, '["B11", "B12"]') == ["B11", "B12"]

    def test_invalid_json_raises_click_error(self):
        with pytest.raises(click.BadParameter):
            pi.parse_list_string(None, None, "not-json")


class TestScaleInputToReference:
    def test_identity_returns_unit_scale(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        scale = pi.scale_input_to_reference(ref.copy(), ref)
        assert scale == pytest.approx(1.0)

    def test_scaled_input_recovered(self):
        ref = np.array([[1.0, 2.0], [3.0, 4.0]])
        scaled = ref * 0.5
        recovered = pi.scale_input_to_reference(scaled, ref)
        assert recovered == pytest.approx(2.0)

    def test_nan_pixels_ignored(self):
        ref = np.array([[1.0, 2.0], [np.nan, 4.0]])
        inp = np.array([[1.0, 2.0], [3.0, 4.0]])
        scale = pi.scale_input_to_reference(inp, ref)
        assert math.isfinite(scale)


class TestNormalizeInputs:
    def test_normalization_factor_applied_and_mean_removed(self):
        data = np.ones((2, 4, 4), dtype=np.float32) * 10.0
        out = pi.normalize_inputs(data.copy(), normalization=0.1)
        # After scaling by 0.1 -> mean 1.0, subtract mean -> all zeros.
        assert np.allclose(out, 0.0)

    def test_multiband_mean_subtraction(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(3, 8, 8)).astype(np.float32) + 5.0
        out = pi.normalize_inputs(data.copy(), normalization=1.0)
        per_pixel_mean = np.nanmean(out, axis=0)
        assert np.allclose(per_pixel_mean, 0.0, atol=1e-5)


class TestMatchedFilter:
    def test_identity_template_scales_sum(self):
        data = np.ones((2, 4, 4), dtype=np.float32)
        cov = np.ones_like(data)
        template = np.array([1.0, 1.0])
        result = pi.matched_filter(data, cov, template)
        # sum(data/cov * template) = 2, sum(template**2/cov) = 2 => 1.0 everywhere
        assert np.allclose(result, 1.0)


# ---------------------------------------------------------------------------
# File-output helpers (use tmp_path)
# ---------------------------------------------------------------------------

def _simple_transform() -> "pi.affine.Affine":
    import affine
    return affine.Affine(0.0001, 0.0, -3.7, 0.0, -0.0001, 40.3)


class TestCreateOrtho:
    def test_2d_input_writes_single_band(self, tmp_path: Path):
        data = np.ones((64, 64), dtype=np.float32)
        out = tmp_path / "single.tif"
        pi.create_ortho(data, _simple_transform(), str(out))
        with rasterio.open(out) as ds:
            assert ds.count == 1
            assert ds.dtypes[0] == "float32"
            # The single-resolution policy: no overviews, by design.
            assert ds.overviews(1) == []

    def test_3d_input_writes_n_bands(self, tmp_path: Path):
        data = np.ones((3, 32, 32), dtype=np.uint8) * 200
        out = tmp_path / "rgb.tif"
        pi.create_ortho(data, _simple_transform(), str(out))
        with rasterio.open(out) as ds:
            assert ds.count == 3
            assert ds.dtypes == ("uint8", "uint8", "uint8")

    def test_uint8_dtype_preserved(self, tmp_path: Path):
        """Regression guard: old create_ortho hardcoded float32, corrupting uint8 RGB."""
        data = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = tmp_path / "u8.tif"
        pi.create_ortho(data, _simple_transform(), str(out))
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "uint8"
            assert np.array_equal(ds.read(1), data)

    def test_invalid_ndim_raises(self, tmp_path: Path):
        out = tmp_path / "bad.tif"
        with pytest.raises(ValueError):
            pi.create_ortho(np.ones(5, dtype=np.float32), _simple_transform(), str(out))
        with pytest.raises(ValueError):
            pi.create_ortho(np.ones((2, 2, 2, 2), dtype=np.float32), _simple_transform(), str(out))


class TestCreateHeatmapCog:
    def test_produces_rgb_uint8_with_mask_applied(self, tmp_path: Path):
        grayscale = np.linspace(-2.0, 2.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        gray_path = tmp_path / "gray.tif"
        pi.create_ortho(grayscale, _simple_transform(), str(gray_path))

        mask = np.zeros_like(grayscale, dtype=np.uint8)
        mask[:32, :] = 1

        rgb_path = tmp_path / "rgb.tif"
        pi.create_heatmap_cog(str(gray_path), str(rgb_path), mask, input_min=-2.0, input_max=2.0)

        with rasterio.open(rgb_path) as ds:
            assert ds.count == 3
            assert ds.dtypes == ("uint8", "uint8", "uint8")
            assert ds.overviews(1) == []
            # Masked rows must be nodata (0); unmasked must have some colorized content.
            for band in range(1, 4):
                arr = ds.read(band)
                assert np.all(arr[32:, :] == 0)
                assert arr[:32, :].max() > 0


class TestGenerateColormap:
    def test_writes_nonempty_png(self, tmp_path: Path, monkeypatch):
        import matplotlib.pyplot as plt

        monkeypatch.setattr(pi, "ASSETS_OUT", str(tmp_path))
        out = pi.generate_colormap(-2.0, 2.0, "legend.png", plt.cm.RdBu_r)
        p = Path(out)
        assert p.exists()
        assert p.stat().st_size > 0


class TestWriteTimeSignalJson:
    def test_payload_shape_matches_aggregator_contract(self, tmp_path: Path, monkeypatch):
        """
        aggregate_signals.extract_datetime reads ``data["datetime"]`` at the
        root first; this test pins that contract at the producer side.
        """
        monkeypatch.setattr(pi, "ASSETS_OUT", str(tmp_path))
        item_dict = {"properties": {"datetime": "2023-01-06T11:20:00Z"}}
        values = [{"date": "2023-01-06T11:20:00Z", "value": 0.12}]
        pi.write_time_signal_json(item_dict, values, "abc")

        payload = json.loads((tmp_path / "abc_time_signal.json").read_text())
        assert payload["datetime"] == "2023-01-06T11:20:00Z"
        assert payload["values"] == values


class TestGetCatalogUrl:
    def test_missing_env_exits(self, monkeypatch):
        monkeypatch.delenv("CATALOG_URL", raising=False)
        with pytest.raises(SystemExit) as exc:
            pi.get_catalog_url()
        assert exc.value.code == 1

    def test_present_env_is_returned(self, monkeypatch):
        monkeypatch.setenv("CATALOG_URL", "https://example.com/stac")
        assert pi.get_catalog_url() == "https://example.com/stac"


class TestEnsureOutputDirectories:
    def test_creates_all_three_directories(self, tmp_path: Path, monkeypatch):
        out_dir = tmp_path / "out"
        monkeypatch.setattr(pi, "OUT_DIR", str(out_dir))
        monkeypatch.setattr(pi, "STAC_ITEMS_OUT", str(out_dir / "stac_items"))
        monkeypatch.setattr(pi, "ASSETS_OUT", str(out_dir / "assets"))
        pi.ensure_output_directories()
        assert (out_dir).is_dir()
        assert (out_dir / "stac_items").is_dir()
        assert (out_dir / "assets").is_dir()

# E2E Tests

Two modes:

| Marker | Network? | S3? | Runs by default? |
|---|---|---|---|
| `e2e_mocked` | No | No | Yes |
| `e2e_real` | Yes (live STAC) | Yes | No, opt-in |

## How `e2e_mocked` stays offline

These tests do not commit any binary fixtures. Every run generates fresh
synthetic GeoTIFFs in `tmp_path` via the `synthetic_l1c_band`,
`synthetic_l2a_scl`, and `synthetic_l2a_visual` fixtures in
[`tests/conftest.py`](../conftest.py), and points STAC item assets at those
files via `file://` URLs. `PyStacClient` and `boto3.Session` are
monkey-patched so no network or AWS call is ever attempted. Adding a new
provider to the mocked matrix means: (a) new branch in
`_build_mocked_pipeline` mapping the provider's asset key names onto the same
synthetic rasters, and (b) a fixture + smoke test mirroring the existing
`e84` / `cdse` / `ed` ones — no fixture files needed.

## Run commands

```bash
# All tests that run without credentials (default)
pytest

# Mocked end-to-end only
pytest -m e2e_mocked

# Real E2E — all providers (skips whichever has no creds set)
pytest -m e2e_real -v

# Real E2E — single provider
pytest -m e2e_real -k e84   -v
pytest -m e2e_real -k cdse  -v
pytest -m e2e_real -k ed    -v

# Everything including real
pytest -m '' -v
```

---

## Provider env var setup

### e84 — Element84 Earth Search

Catalog URL is hardcoded (`https://earth-search.aws.element84.com/v1`).
Only S3 credentials are required.

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-west-2"          # optional, defaults to us-west-2
```

---

### cdse — Copernicus Data Space Ecosystem

Catalog URL is hardcoded (`https://stac.dataspace.copernicus.eu/v1/`).
Only CDSE S3 credentials are required.
Get them from: https://eodata-s3keysmanager.dataspace.copernicus.eu

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

---

### ed — EDA

No hardcoded catalog URL — you must supply it via `CATALOG_URL`.
Same AWS S3 backend as `e84` (requester-pays, `us-west-2`).

```bash
export CATALOG_URL="https://your-eda-stac-endpoint/v1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-west-2"
```

---

## Override test parameters

The suite ships with defaults that cover a well-tested AOI over central Spain.
Override any of them per run:

```bash
export METHANE_E2E_BBOX='[-3.67, 40.23, -3.61, 40.29]'
export METHANE_E2E_START='2025-01-01T00:00:00Z'
export METHANE_E2E_END='2025-01-31T23:59:59Z'
export METHANE_E2E_CLOUD_COVER='10'
export METHANE_E2E_LIMIT='10'         # set to 1 or 2 for faster runs
```

---

## Troubleshooting

- **Provider skipped** — check the skip message; it lists the exact missing env vars.
- **403 from S3** — credentials are invalid or expired. For e84, verify with `aws sts get-caller-identity`.
- **Empty search results** — no items matched the AOI/date/cloud filter. Lower `METHANE_E2E_CLOUD_COVER` or widen the date range.
- **Slow runs** — set `METHANE_E2E_LIMIT=1` to process one item only.

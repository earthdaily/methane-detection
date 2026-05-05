# Methane detection (Sentinel-2 STAC pipeline)

Python CLIs that search a STAC catalog for Sentinel-2 scenes, run a matched-filter methane enhancement on each L1C item, optionally aggregate per-scene time signals, and write artifacts under `out/`. The tools are packaged in a Docker image for workflows (e.g. Argo) or local runs.

## What is in this repo

| File | Role |
|------|------|
| `stac_search.py` | Queries the catalog for L1C (and paired L2A) items; prints a **JSON array** to **stdout** (logs go to stderr). |
| `process_item.py` | Processes **one** L1C item: reads bands from cloud storage, applies L2A cloud masking when paired, runs the methane matched filter, writes GeoTIFFs/PNGs/JSON under `out/`. |
| `aggregate_signals.py` | Scans `out/assets` for `*_time_signal.json` and writes `out/signals/items_time_signal.json` with per-datetime summary stats. |
| `run_pipeline.py` | Runs the complete flow: STAC search → per-item processing → signal aggregation. |
| `providers.py` | Single source of truth for all provider config (catalog URL, band keys, S3 endpoint, filter style). Add a new provider here — nothing else needs to change. |
| `app-package.cwl` | EOAP/CWL Workflow package for deploying the complete flow as an OGC API Processes-style application. |
| `Dockerfile` | Python 3.12 image with GDAL/rasterio system deps and pinned deps from `requirements.txt`. |
| `requirements.txt` | Locked Python dependencies used by the image and for local `pip install`. |

Typical flow: **search → many parallel `process_item` runs → aggregate**.

---

## Supported STAC providers

Pass `--stac-provider` to any script. The default is `e84`.

| Provider | `--stac-provider` | Catalog URL | S3 backend | Filter style |
|---|---|---|---|---|
| Element84 Earth Search | `e84` | `https://earth-search.aws.element84.com/v1` (hardcoded) | AWS S3 (requester-pays) | STAC Query extension |
| Copernicus Data Space | `cdse` | `https://stac.dataspace.copernicus.eu/v1/` (hardcoded) | CDSE S3 | CQL2 |
| ED | `ed` | Must be set via `CATALOG_URL` env var | AWS S3 (requester-pays) | CQL2 |

Provider config lives in `providers.py`. Adding a new provider (e.g. Microsoft Planetary Computer) = one new dict entry there, nothing else.

---

## Credentials setup

### e84 — Element84

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-west-2"          # optional, defaults to us-west-2
```

### cdse — Copernicus Data Space

Get keys from: https://eodata-s3keysmanager.dataspace.copernicus.eu

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### ed — EDA

Same AWS S3 backend as `e84` (requester-pays, `us-west-2`). Only difference is the catalog URL, which must be supplied:

```bash
export CATALOG_URL="https://ed-stac-endpoint/v1/stac"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-west-2"
```

---

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### `stac_search.py`

Searches L1C and paired L2A collections over a bbox and time range, deduplicates near-duplicate scenes, and pairs L2A ids where overlap and acquisition time match.

Stdout is a JSON list: `[{"sentinel-2-l1c": "<id>", "sentinel-2-l2a": "<id or null>"}, ...]`

```bash
python stac_search.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start-datetime 2025-01-01T00:00:00Z \
  --end-datetime   2025-01-31T23:59:59Z \
  --cloud-cover 10 \
  --limit 5 \
  --stac-provider e84
```

For `cdse` or `ed`, omit `CATALOG_URL` (cdse) or set it (ed) and swap the provider flag:

```bash
# cdse
python stac_search.py --stac-provider cdse --bbox '...' --start-datetime ... --end-datetime ...

# ed
export CATALOG_URL="https://your-eda-stac-endpoint/v1"
python stac_search.py --stac-provider ed --bbox '...' --start-datetime ... --end-datetime ...
```

### `process_item.py`

Processes a **single** L1C item. Writes under `out/stac_items/` and `out/assets/`.

Required flags:

- `--bbox` — JSON `[west, south, east, north]`
- `--collection` — STAC collection name (e.g. `sentinel-2-l1c`)
- `--l1c-id` — L1C item id
- `--stac-provider` — `e84` / `cdse` / `ed`

Optional flags:

- `--l2a-id` — paired L2A item for SCL-based masking and RGB output
- `--download-bands-list` — JSON list of asset keys (provider default used if omitted)
- `--skip-viz` — skip matplotlib PNGs
- `--skip-colorized` — skip colorized heatmap COG
- `--skip-overviews` — single-resolution GeoTIFF outputs
- `--auto-res` — derive WGS84 pixel size from the first L1C band instead of using `METHANE_TARGET_RES`

```bash
# e84
python process_item.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --collection sentinel-2-l1c \
  --l1c-id S2A_MSIL1C_20250115T105021_N0510_R051_T30TVK_20250115T123456 \
  --l2a-id S2A_MSIL2A_20250115T105021_N0510_R051_T30TVK_20250115T234567 \
  --stac-provider e84

# cdse
python process_item.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --collection sentinel-2-l1c \
  --l1c-id S2C_MSIL1C_20250130T111341_N0511_R137_T30TVK_20250206T144330 \
  --stac-provider cdse
```

### `aggregate_signals.py`

After one or more `process_item` runs have written `out/assets/*_time_signal.json`:

```bash
python aggregate_signals.py \
  --assets-dir out/assets \
  --signals-dir out/signals
```

Writes `out/signals/items_time_signal.json`. If no signal files exist, writes an empty aggregate and logs a warning.

### `run_pipeline.py`

Runs the full search → process → aggregate flow in one command. This is the executable used by `app-package.cwl`.

```bash
# e84
python run_pipeline.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start-datetime 2025-01-01T00:00:00Z \
  --end-datetime   2025-01-31T23:59:59Z \
  --limit 5 \
  --stac-provider e84

# cdse
python run_pipeline.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start-datetime 2025-01-01T00:00:00Z \
  --end-datetime   2025-01-31T23:59:59Z \
  --stac-provider cdse

# ed
export CATALOG_URL="https://ed-stac-endpoint/v1/stac"
python run_pipeline.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start-datetime 2025-01-01T00:00:00Z \
  --end-datetime   2025-01-31T23:59:59Z \
  --stac-provider ed
```

### `app-package.cwl`

```bash
cwltool app-package.cwl \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start-datetime 2025-01-01T00:00:00Z \
  --end-datetime   2025-01-31T23:59:59Z \
  --limit 5 \
  --stac-provider e84
```

#### Testing the CWL with a locally built image

The CWL pins `dockerPull: docker.io/earthdaily/methane-detection:vX.Y.Z`, so by default cwltool fetches the published image. To validate a local build before publishing, tag your build with that exact name and disable the registry pull:

```bash
docker build -t methane-detection:test .
docker tag methane-detection:test docker.io/earthdaily/methane-detection:vX.Y.Z   # match the pin

# Credentials must be exported in the calling shell; cwltool needs to be told which
# env vars to forward into the container.
export AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." AWS_REGION="us-west-2"

cwltool --disable-pull \
  --preserve-environment AWS_ACCESS_KEY_ID \
  --preserve-environment AWS_SECRET_ACCESS_KEY \
  --preserve-environment AWS_REGION \
  --outdir out-e84 \
  app-package.cwl --stac-provider e84 --bbox '[-3.67, 40.23, -3.61, 40.29]' \
    --start-datetime 2025-01-01T00:00:00Z --end-datetime 2025-01-31T23:59:59Z --limit 1
```

For CDSE, re-export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` with the keys from [eodata-s3keysmanager](https://eodata-s3keysmanager.dataspace.copernicus.eu) and run again with `--stac-provider cdse --outdir out-cdse`.

---

## Build and run with Docker

```bash
docker build -t methane-detection:latest .
```

```bash
# e84
docker run --rm \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  -v "$(pwd)/out:/app/out" \
  methane-detection:latest \
  python /app/run_pipeline.py \
    --bbox '[-3.67, 40.23, -3.61, 40.29]' \
    --start-datetime 2025-01-01T00:00:00Z \
    --end-datetime   2025-01-31T23:59:59Z \
    --stac-provider e84

# cdse
docker run --rm \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -v "$(pwd)/out:/app/out" \
  methane-detection:latest \
  python /app/run_pipeline.py \
    --bbox '[-3.67, 40.23, -3.61, 40.29]' \
    --start-datetime 2025-01-01T00:00:00Z \
    --end-datetime   2025-01-31T23:59:59Z \
    --stac-provider cdse
```

---

## Tests

### Default (no credentials needed)

```bash
pytest                    # unit + mocked e2e (default)
pytest tests/unit         # unit only
pytest tests/integration  # local GDAL/rasterio, no network
pytest -m e2e_mocked      # full mocked pipeline (synthetic rasters + mocked STAC)
```

### Real E2E (hits live STAC + S3)

Tests are parametrized over all three providers. Each provider auto-skips if its required env vars are missing — no red failures.

```bash
# All providers at once (skips whichever has no creds)
pytest -m e2e_real -v

# Single provider
pytest -m e2e_real -k e84   -v
pytest -m e2e_real -k cdse  -v
pytest -m e2e_real -k ed    -v
```

Override test parameters:

```bash
export METHANE_E2E_BBOX='[-3.67, 40.23, -3.61, 40.29]'
export METHANE_E2E_START='2025-01-01T00:00:00Z'
export METHANE_E2E_END='2025-01-31T23:59:59Z'
export METHANE_E2E_CLOUD_COVER='10'
export METHANE_E2E_LIMIT='2'           # set low for faster runs
```

See [`tests/e2e/README.md`](tests/e2e/README.md) for full details.

---

## Troubleshooting

- **Provider skipped in real E2E** — the skip message lists the exact missing env vars.
- **`CATALOG_URL` not set for `ed`** — `ed` has no hardcoded default; the script exits with an error until it is set.
- **403 from S3 (e84)** — requester-pays requires valid credentials. Verify with `aws sts get-caller-identity`.
- **403 from S3 (cdse/ed)** — check `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are correct.
- **Empty search results** — no items matched the AOI/date/cloud filter. Lower `--cloud-cover` or widen the date range.
- **Rasterio / GDAL errors in Docker** — rebuild after changing `requirements.txt`.

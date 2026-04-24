# Methane detection (Sentinel-2 STAC pipeline)

Python CLIs that search a STAC catalog for Sentinel-2 scenes, run a matched-filter methane enhancement on each L1C item, optionally aggregate per-scene time signals, and write artifacts under `out/`. The tools are packaged in a Docker image for workflows (for example Argo) or local runs.

## What is in this repo

| File | Role |
|------|------|
| `stac_search.py` | Queries the catalog for L1C (and paired L2A) items; prints a **JSON array** to **stdout** (logs go to stderr). |
| `process_item.py` | Processes **one** L1C item: reads bands from cloud assets (AWS requester-pays), applies L2A cloud masking when paired, runs the methane matched filter, writes GeoTIFFs/PNGs/JSON under `out/`. |
| `aggregate_signals.py` | Scans `out/assets` for `*_time_signal.json` and writes `out/signals/items_time_signal.json` with per-datetime summary stats. |
| `run_pipeline.py` | Runs the complete flow: STAC search, per-item processing, and signal aggregation. |
| `app-package.cwl` | EOAP/CWL Workflow package for deploying the complete flow as an OGC API Processes-style application. |
| `Dockerfile` | Python 3.12 image with GDAL/rasterio system deps and pinned deps from `requirements.txt`. |
| `requirements.txt` | Locked Python dependencies used by the image and for local `pip install`. |

Typical flow: **search → many parallel `process_item` runs → aggregate** (aggregate only if you produced `*_time_signal.json` files).

## Prerequisites

- **Docker** (for the image workflow), or Python 3.12+ with GDAL if you run scripts on the host.
- **`CATALOG_URL`**: base URL of a STAC API that exposes Sentinel-2 L1C/L2A collections compatible with this code (same collection IDs and asset layout you configure).
- **AWS credentials** for `process_item.py`: Sentinel-2 on AWS is accessed with **requester pays** (`AWSSession(..., requester_pays=True)` in code). Configure credentials the usual way (`~/.aws/credentials`, environment variables, or IAM role in Kubernetes).

## Build the Docker image

From the repository root (where the `Dockerfile` lives):

```bash
docker build -t methane-detection:latest .
```

The image installs `requirements.txt`, then copies `stac_search.py`, `process_item.py`, `aggregate_signals.py`, and `run_pipeline.py` into `/app`. There is no default `CMD`; you invoke the scripts explicitly.

To mount outputs and pass config:

```bash
docker run --rm \
  -e CATALOG_URL="https://your-stac-api.example.com/" \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
  -v "$(pwd)/out:/app/out" \
  methane-detection:latest \
  python /app/stac_search.py --help
```

Adjust env vars and volume mount paths as needed. The `.dockerignore` file excludes local `out/`, virtualenvs, secrets, and other non-runtime paths from the build context.

## Run the scripts locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export CATALOG_URL="https://your-stac-api.example.com/"
python stac_search.py --help
python process_item.py --help
python aggregate_signals.py --help
python run_pipeline.py --help
```

You still need GDAL/rasterio-compatible system libraries on the host (the Dockerfile shows the Debian packages used in the container).

### `stac_search.py`

Searches **primary** (`--collection1`, default `sentinel-2-l1c`) and **secondary** (`--collection2`, default `sentinel-2-l2a`) collections over a bbox and time range, deduplicates near-duplicate scenes, and pairs L2A ids where overlap and acquisition time match. **Stdout** is a JSON list like:

`[{"sentinel-2-l1c": "<id>", "sentinel-2-l2a": "<id or null>"}, ...]`

**Catalog URL**: set `CATALOG_URL` or pass `--catalog_url`.

Useful options (see `--help` for all):

- `--bbox` — JSON list `[west, south, east, north]` (default bbox is built-in for demos).
- `--start_datetime` / `--end_datetime` — ISO 8601 range.
- `--cloud_cover` — max `eo:cloud_cover` (requires STAC Query extension support on the server).
- `--limit` — max L1C items; L2A search uses a higher internal limit for pairing.

Example:

```bash
export CATALOG_URL="https://your-stac-api.example.com/"
python stac_search.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start_datetime 2023-01-01T00:00:00Z \
  --end_datetime 2023-01-24T23:59:59Z \
  --limit 5 \
  > items.json
```

### `process_item.py`

Processes a **single** L1C item. **Requires** `CATALOG_URL`. Writes under:

- `out/stac_items/` — item metadata
- `out/assets/` — rasters, plots, and per-item `*_time_signal.json` when signal processing succeeds

Required arguments:

- `--bbox` — JSON `[west, south, east, north]` (the pipeline expands this internally for reads).
- `--collection` — STAC collection name for the L1C item (for example `sentinel-2-l1c`).
- `--l1c-id` — L1C item id.

Optional:

- `--l2a-id` — paired L2A item for SCL-based masking and RGB (omit if none).
- `--download_bands_list` — JSON list of asset keys (default `["B11.jp2", "B12.jp2"]`).
- `--skip-viz` — skip matplotlib PNGs and legend.
- `--skip-colorized` — skip colorized heatmap COG.
- `--skip-overviews` — single-resolution GeoTIFF outputs.
- **`METHANE_TARGET_RES`** — JSON array of two positive floats, WGS84 degrees per pixel, e.g. `[0.00018, 0.00018]` (default matches the previous built-in resolution). Ignored when using `--auto-res` for the chosen grid resolution (snap/origin logic unchanged).
- **`--auto-res`** — derive WGS84 pixel size from the first L1C band (`rasterio.open` once), then run the same snapped `compute_target_grid` as fixed mode. Default off preserves prior behavior when `METHANE_TARGET_RES` is unset.

Example (after you have ids from search or elsewhere):

```bash
export CATALOG_URL="https://your-stac-api.example.com/"
python process_item.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --collection sentinel-2-l1c \
  --l1c-id S2A_MSIL1C_20230115T105021_N0510_R051_T30TVK_20230115T123456 \
  --l2a-id S2A_MSIL2A_20230115T105021_N0510_R051_T30TVK_20230115T234567
```

### `aggregate_signals.py`

After one or more `process_item` runs have written `out/assets/*_time_signal.json`, combine them:

```bash
python aggregate_signals.py \
  --assets-dir out/assets \
  --signals-dir out/signals
```

This creates `out/signals/items_time_signal.json` (or an empty structure if no matching files exist). If there are no `*_time_signal.json` files, the command still writes an empty aggregate file and logs a warning.

### `run_pipeline.py`

Runs the full search → process → aggregate flow in one command. This is the executable used by `app-package.cwl`.

```bash
export CATALOG_URL="https://your-stac-api.example.com/"
python run_pipeline.py \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start_datetime 2023-01-01T00:00:00Z \
  --end_datetime 2023-01-24T23:59:59Z \
  --limit 5
```

### `app-package.cwl`

The EOAP package is a CWL `Workflow` that exposes the complete repository capability through `run_pipeline.py` and writes the `out/` directory as the workflow output.

```bash
cwltool app-package.cwl \
  --bbox '[-3.67, 40.23, -3.61, 40.29]' \
  --start_datetime 2023-01-01T00:00:00Z \
  --end_datetime 2023-01-24T23:59:59Z \
  --limit 5 \
  --catalog_url https://earth-search.aws.element84.com/v1
```

## End-to-end shell sketch

This is illustrative: parse `items.json` with `jq` or your orchestrator to fan out `process_item` per element.

```bash
export CATALOG_URL="https://your-stac-api.example.com/"
python stac_search.py --limit 2 > items.json
# For each object in items.json, run process_item.py with .["sentinel-2-l1c"], .["sentinel-2-l2a"], etc.
python aggregate_signals.py
```

## Troubleshooting

- **`CATALOG_URL` not set** — both search and process exit with an error until it is defined.
- **Rasterio / GDAL errors in Docker** — rebuild after changing `requirements.txt`; the image pins `GDAL` and `rasterio` to versions that expect the Debian `libgdal-dev` in the Dockerfile.
- **Empty or failed reads from AWS** — confirm AWS credentials, region, and that your account accepts **requester-pays** charges for the Sentinel-2 bucket you use.

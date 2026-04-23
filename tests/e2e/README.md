# E2E Tests

Two modes:

| Marker | Network? | AWS? | Runs by default? |
|---|---|---|---|
| `e2e_mocked` | No | No | Yes |
| `e2e_real`   | Yes (live STAC) | Yes (requester-pays S3) | No, opt-in |

## Command cheat sheet

```bash
# Everything that runs without network (default)
pytest

# Just unit
pytest tests/unit

# Just integration (synthetic GeoTIFF + mocked STAC)
pytest tests/integration

# Mocked end-to-end (synthetic pipeline)
pytest -m e2e_mocked

# Live end-to-end (see env vars below)
pytest -m e2e_real

# Everything, including real
pytest -m ''
```

## Running the real E2E suite

The real tests chain the production topology end-to-end:

```
stac_search.py  ->  process_item.py (per item)  ->  aggregate_signals.py
```

They expect the **same env vars `process_item.py` itself reads at runtime**.

### Required env vars

```bash
export CATALOG_URL="https://earth-search.aws.element84.com/v1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-west-2"           # Sentinel-2 L1C/L2A bucket region
# Optional, normally picked up from AWS_PROFILE or the default chain
# export AWS_SESSION_TOKEN="..."
```

If any required var is missing, the real tests skip cleanly with a message
listing what's missing — no red failures.

### Test parameters (well-tested defaults)

The suite ships with the exact parameters used for routine manual QA of
the pipeline. No env vars are required for these to work — they're the
hard-coded defaults:

| Parameter | Default |
|---|---|
| bbox | `[-3.67, 40.23, -3.61, 40.29]` |
| start_datetime | `2023-01-01T00:00:00Z` |
| end_datetime | `2023-01-31T23:59:59Z` |
| cloud_cover | `10` |
| limit | `10` |

You can override any of them per-run if needed:

```bash
export METHANE_E2E_BBOX='[...]'
export METHANE_E2E_START='2023-02-01T00:00:00Z'
export METHANE_E2E_END='2023-02-28T23:59:59Z'
export METHANE_E2E_CLOUD_COVER='20'
export METHANE_E2E_LIMIT='5'
```

### Running

```bash
pytest -m e2e_real -v
```

Each test spawns subprocesses and writes outputs under `pytest`'s
`tmp_path`, so repeated runs are isolated. First-run times of a few
minutes per test are normal (Sentinel-2 tiles are ~100 MB each).

### Troubleshooting

- **403 from S3** — requester-pays requires authenticated credentials. `AWSSession(..., requester_pays=True)` handles the header, but the creds still need to be valid. Verify `aws sts get-caller-identity` first.
- **Empty search payload** — no L1C items matched the AOI/date/cloud filter. Either expand the window or lower the cloud threshold.
- **Hit rate limits or slow** — set `METHANE_E2E_LIMIT=1` or `METHANE_E2E_LIMIT=2` to process fewer items per run.

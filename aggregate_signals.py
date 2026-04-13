"""
Aggregate per-item time_signal JSON files into a single items_time_signal.json.

Reads all *_time_signal.json from an assets directory and writes
out/signals/items_time_signal.json with structure:
  {
    "data": {
      "values": [
        {"datetime": "<iso>", "min": <float>, "q1": <float>, "median": <float>, "q3": <float>, "max": <float>, "mean": <float>, "count": <int>},
        ...
      ]
    }
  }
  Each record summarizes per-pixel signal values for that item (from values[].value).
"""
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUFFIX = "_time_signal.json"
OUTPUT_FILENAME = "items_time_signal.json"
EMPTY_OUTPUT = {"data": {"values": []}}


def compute_boxplot_stats(signal_list: list[float]) -> dict | None:
    """Compute min, q1, median, q3, max, mean, count from signal list. Floats rounded to 2 decimals."""
    if not signal_list:
        return None
    n = len(signal_list)
    if n == 1:
        x = round(signal_list[0], 2)
        return {
            "min": x,
            "q1": x,
            "median": x,
            "q3": x,
            "max": x,
            "mean": x,
            "count": n,
        }
    q1, median, q3 = statistics.quantiles(signal_list, n=4)
    return {
        "min": round(min(signal_list), 2),
        "q1": round(q1, 2),
        "median": round(median, 2),
        "q3": round(q3, 2),
        "max": round(max(signal_list), 2),
        "mean": round(statistics.mean(signal_list), 2),
        "count": n,
    }


def iter_signal_files(assets_path: Path) -> list[Path]:
    """Return all per-item signal files in deterministic order."""
    pattern = f"*{SUFFIX}"
    return sorted(assets_path.glob(pattern))


def extract_signal_values(values: list[dict[str, Any]]) -> list[float]:
    """Extract numeric signal values from the per-item payload."""
    signal_list = []
    for value in values:
        raw_value = value.get("value")
        if raw_value is None:
            continue
        try:
            signal_list.append(float(raw_value))
        except (TypeError, ValueError):
            logger.warning(f"Skipping non-numeric signal value: {raw_value!r}")
    return signal_list


def extract_datetime(data: dict[str, Any], values: list[dict[str, Any]]) -> str | None:
    """Extract the item timestamp from either the root record or the first value."""
    if data.get("datetime"):
        return data["datetime"]
    if values:
        return values[0].get("date")
    return None


def record_from_data(data: dict[str, Any]) -> dict | None:
    """Build one output record from a per-item time_signal.json with boxplot stats."""
    values = data.get("values") or []
    signal_list = extract_signal_values(values)
    dt = extract_datetime(data, values)
    if not dt:
        return None
    stats = compute_boxplot_stats(signal_list)
    if stats is None:
        return None
    return {
        "datetime": dt,
        **stats,
    }


def load_record(path: Path) -> dict | None:
    """Load one signal file and convert it into an aggregate record."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Skipping {path}: {e}")
        return None

    record = record_from_data(data)
    if record is None:
        logger.warning(f"Skipping {path}: missing datetime or valid signal values")
    return record


def write_aggregated_output(signals_path: Path, records: list[dict[str, Any]]) -> Path:
    """Write aggregate boxplot statistics to disk."""
    signals_path.mkdir(parents=True, exist_ok=True)
    out_file = signals_path / OUTPUT_FILENAME
    with open(out_file, "w") as f:
        json.dump({"data": {"values": records}}, f, indent=2)
    return out_file


@click.command()
@click.option(
    "--assets-dir",
    type=click.Path(exists=True, file_okay=False),
    default="out/assets",
    help="Directory containing *_time_signal.json files",
)
@click.option(
    "--signals-dir",
    type=click.Path(),
    default="out/signals",
    help="Output directory for items_time_signal.json",
)
def main(assets_dir: str, signals_dir: str) -> None:
    """Aggregate *_time_signal.json files into signals/items_time_signal.json."""
    assets_path = Path(assets_dir)
    signals_path = Path(signals_dir)
    files = iter_signal_files(assets_path)

    if not files:
        logger.warning(f"No *{SUFFIX} files found in {assets_path}")
        signals_path.mkdir(parents=True, exist_ok=True)
        out_file = signals_path / OUTPUT_FILENAME
        with open(out_file, "w") as f:
            json.dump(EMPTY_OUTPUT, f, indent=2)
        logger.info(f"Wrote empty data to {out_file}")
        return

    records = []
    for path in files:
        record = load_record(path)
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r.get("datetime") or "")
    out_file = write_aggregated_output(signals_path, records)

    logger.info(f"Aggregated {len(records)} signals to {out_file}")


if __name__ == "__main__":
    main()

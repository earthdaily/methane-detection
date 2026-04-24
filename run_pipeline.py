"""
Run the complete methane-detection pipeline.

The pipeline searches for Sentinel-2 L1C scenes, processes each matching item,
and aggregates per-item time-signal JSON files into one summary output.
"""
import json
import logging
import os
import sys

import click

import aggregate_signals
import process_item
import stac_search


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_process_item(
    bbox: list[float],
    collection: str,
    l1c_id: str,
    l2a_id: str | None,
    download_bands_list: list[str],
    skip_viz: bool,
    skip_colorized: bool,
    skip_overviews: bool,
) -> None:
    """Invoke the per-item Click command with the same CLI surface as CWL."""
    args = [
        "--bbox",
        json.dumps(bbox),
        "--collection",
        collection,
        "--l1c-id",
        l1c_id,
        "--download_bands_list",
        json.dumps(download_bands_list),
    ]
    if l2a_id:
        args.extend(["--l2a-id", l2a_id])
    if skip_viz:
        args.append("--skip-viz")
    if skip_colorized:
        args.append("--skip-colorized")
    if skip_overviews:
        args.append("--skip-overviews")

    process_item.main(args=args, standalone_mode=False)


@click.command()
@click.option(
    "--bbox",
    callback=stac_search.parse_list_string,
    required=True,
    help="Bounding box as JSON list [west, south, east, north].",
)
@click.option(
    "--start_datetime",
    default=stac_search.DEFAULT_START_DATETIME,
    show_default=True,
    help="Start datetime in ISO 8601 format.",
)
@click.option(
    "--end_datetime",
    default=stac_search.DEFAULT_END_DATETIME,
    show_default=True,
    help="End datetime in ISO 8601 format.",
)
@click.option(
    "--collection1",
    default=stac_search.DEFAULT_PRIMARY_COLLECTION,
    show_default=True,
    help="Primary STAC collection to search for L1C scenes.",
)
@click.option(
    "--collection2",
    default=stac_search.DEFAULT_SECONDARY_COLLECTION,
    show_default=True,
    help="Secondary STAC collection used for L2A pairing.",
)
@click.option(
    "--cloud_cover",
    callback=stac_search.parse_optional_float,
    default=str(stac_search.DEFAULT_CLOUD_COVER),
    show_default=True,
    help="Max eo:cloud_cover value. Use empty string to disable.",
)
@click.option(
    "--limit",
    callback=stac_search.parse_optional_int,
    default=str(stac_search.DEFAULT_LIMIT),
    show_default=True,
    help="Max number of L1C items to process. Use empty string for no limit.",
)
@click.option(
    "--download_bands_list",
    callback=process_item.parse_list_string,
    default='["B11.jp2", "B12.jp2"]',
    show_default=True,
    help="JSON list of band asset keys to process.",
)
@click.option("--skip-viz", is_flag=True, default=False, help="Skip PNG visual outputs.")
@click.option(
    "--skip-colorized",
    is_flag=True,
    default=False,
    help="Skip colorized methane heatmap COG outputs.",
)
@click.option(
    "--skip-overviews",
    is_flag=True,
    default=False,
    help="Request single-resolution GeoTIFF outputs.",
)
@click.option(
    "--catalog_url",
    required=False,
    default=None,
    help="STAC API endpoint. Defaults to CATALOG_URL env var.",
)
def main(
    bbox: list[float],
    start_datetime: str,
    end_datetime: str,
    collection1: str,
    collection2: str,
    cloud_cover: float | None,
    limit: int | None,
    download_bands_list: list[str],
    skip_viz: bool,
    skip_colorized: bool,
    skip_overviews: bool,
    catalog_url: str | None,
) -> None:
    """Search, process all matching items, and aggregate time-signal outputs."""
    catalog_url = catalog_url or os.getenv("CATALOG_URL", "")
    if not catalog_url:
        logger.error("No catalog URL provided and CATALOG_URL env var not set")
        sys.exit(1)
    os.environ["CATALOG_URL"] = catalog_url

    try:
        l1c_items = stac_search.deduplicate_items(
            stac_search.search_stac(
                bbox=bbox,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                collection=collection1,
                catalog_url=catalog_url,
                cloud_cover=cloud_cover,
                limit=limit,
            )
        )
        l2a_items = stac_search.deduplicate_items(
            stac_search.search_stac(
                bbox=bbox,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                collection=collection2,
                catalog_url=catalog_url,
                cloud_cover=cloud_cover,
                limit=stac_search.get_l2a_pairing_limit(limit),
            )
        )
        payload = stac_search.build_output_payload(l1c_items, l2a_items)
        logger.info(f"Processing {len(payload)} L1C item(s)")

        processed_item_ids = []
        for item in payload:
            l1c_id = item["sentinel-2-l1c"]
            run_process_item(
                bbox=bbox,
                collection=collection1,
                l1c_id=l1c_id,
                l2a_id=item.get("sentinel-2-l2a"),
                download_bands_list=download_bands_list,
                skip_viz=skip_viz,
                skip_colorized=skip_colorized,
                skip_overviews=skip_overviews,
            )
            processed_item_ids.append(l1c_id)

        process_item.write_stac_catalog(processed_item_ids)
        aggregate_signals.main(
            args=["--assets-dir", process_item.ASSETS_OUT, "--signals-dir", "out/signals"],
            standalone_mode=False,
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

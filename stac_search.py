"""
STAC Search CLI

This script performs a STAC search based on bounding box and datetime range,
outputting a JSON array of item IDs to stdout for use in Argo workflow fan-out.
"""
import json
import logging
import os
import sys
from shapely import geometry

import pystac
import click
from pystac_client import Client as PyStacClient

# Configure logging to stderr so stdout is clean for JSON output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Module-level defaults — used both in the CLI and as documentation
DEFAULT_BBOX = [-3.67, 40.23, -3.61, 40.29]
DEFAULT_PRIMARY_COLLECTION = "sentinel-2-l1c"
DEFAULT_SECONDARY_COLLECTION = "sentinel-2-l2a"
DEFAULT_START_DATETIME = "2023-01-01T00:00:00Z"
DEFAULT_END_DATETIME = "2023-01-24T23:59:59Z"
DEFAULT_CLOUD_COVER = 10.0
DEFAULT_LIMIT = 10
MIN_L2A_PAIRING_LIMIT = 50


def get_l2a_pairing_limit(limit: int | None) -> int | None:
    """Return a broader search limit for L2A pairing to improve match rate."""
    if limit is None:
        return None
    return max(limit * 10, MIN_L2A_PAIRING_LIMIT)


def parse_list_string(ctx, param, value):
    """Parse a JSON string into a Python list."""
    try:
        return json.loads(value)
    except ValueError as e:
        logger.error(e)
        raise click.BadParameter(
            "Could not parse the input as a list", param_hint=value
        )


def parse_optional_float(ctx, param, value):
    """Convert empty string to None, otherwise to float."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise click.BadParameter(f"Expected a number or empty, got {value!r}")


def parse_optional_int(ctx, param, value):
    """Convert empty string to None, otherwise to int."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise click.BadParameter(f"Expected an integer or empty, got {value!r}")


def select_latest_created_item(items: list[pystac.Item]) -> pystac.Item:
    """Select the most recently created item from a set of near-duplicate scenes."""
    return sorted(
        items,
        key=lambda item: item.properties["created"],
        reverse=True,
    )[0]


def deduplicate_items(items: list[pystac.Item]) -> list[pystac.Item]:
    """Remove near-duplicate STAC items acquired within 10 minutes."""
    if len(items) == 0:
        return []

    latest_items = []
    for item in items:
        nearby_items = []
        for candidate in items:
            if abs((item.datetime - candidate.datetime).total_seconds()) < 600:
                nearby_items.append(candidate)
        latest_items.append(select_latest_created_item(nearby_items))

    seen = set()
    seen_add = seen.add
    deduplicated_items = [x for x in latest_items if not (x in seen or seen_add(x))]
    return deduplicated_items


def has_sufficient_overlap(l1c_item: pystac.Item, l2a_item: pystac.Item) -> bool:
    """Check whether the L2A footprint overlaps enough with the L1C scene."""
    l1c_geometry = geometry.box(*l1c_item.bbox)
    l2a_geometry = geometry.box(*l2a_item.bbox)
    overlap_fraction = l1c_geometry.intersection(l2a_geometry).area / l1c_geometry.area
    return overlap_fraction > 0.5


def match_l2a_item_id(l1c_item: pystac.Item, l2a_items: list[pystac.Item]) -> str | None:
    """Find the first matching L2A item for an L1C item."""
    for l2a_item in l2a_items:
        time_diff_seconds = abs((l2a_item.datetime - l1c_item.datetime).total_seconds())
        if has_sufficient_overlap(l1c_item, l2a_item) and time_diff_seconds < 1:
            return l2a_item.id
    return None


def build_output_payload(
    l1c_items: list[pystac.Item],
    l2a_items: list[pystac.Item],
) -> list[dict[str, str | None]]:
    """Build the Argo fan-out payload with optional paired L2A IDs."""
    return [
        {
            "sentinel-2-l1c": l1c_item.id,
            "sentinel-2-l2a": match_l2a_item_id(l1c_item, l2a_items),
        }
        for l1c_item in l1c_items
    ]


def search_stac(
    bbox: list[float],
    start_datetime: str,
    end_datetime: str,
    collection: str,
    catalog_url: str,
    cloud_cover: float | None = None,
    limit: int | None = None,
) -> list[pystac.Item]:
    """
    Search STAC catalog for items matching the given criteria.

    Args:
        bbox: Bounding box [west, south, east, north]
        start_datetime: Start datetime in ISO format
        end_datetime: End datetime in ISO format
        collection: STAC collection to search
        catalog_url: URL of the STAC catalog
        cloud_cover: Optional max cloud cover (eo:cloud_cover<=value). Requires Query extension.
        limit: Optional max number of items to return (passed as max_items).

    Returns:
        List of STAC items matching the search criteria
    """
    logger.info(f"Searching STAC catalog: {catalog_url}")
    logger.info(f"Collection: {collection}")
    logger.info(f"Bbox: {bbox}")
    logger.info(f"Datetime range: {start_datetime} to {end_datetime}")
    if cloud_cover is not None:
        logger.info(f"Cloud cover filter: <={cloud_cover}")
    if limit is not None:
        logger.info(f"Limit (max_items): {limit}")

    client = PyStacClient.open(catalog_url)

    datetime_range = f"{start_datetime}/{end_datetime}"

    search_kw: dict = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": datetime_range,
    }
    if cloud_cover is not None:
        search_kw["query"] = {"eo:cloud_cover": {"lte": cloud_cover}}
    if limit is not None:
        search_kw["max_items"] = limit

    results = client.search(**search_kw)

    items = results.item_collection().items
    logger.info(f"Found {len(items)} items")

    return items


@click.command()
@click.option(
    "--bbox",
    callback=parse_list_string,
    required=False,
    default=json.dumps(DEFAULT_BBOX),
    show_default=True,
    help="Bounding box as JSON list [west, south, east, north]",
)
@click.option(
    "--start_datetime",
    required=False,
    default=DEFAULT_START_DATETIME,
    show_default=True,
    help="Start datetime in ISO format (e.g., 2023-01-01T00:00:00Z)",
)
@click.option(
    "--end_datetime",
    required=False,
    default=DEFAULT_END_DATETIME,
    show_default=True,
    help="End datetime in ISO format (e.g., 2023-12-31T23:59:59Z)",
)
@click.option(
    "--collection1",
    required=False,
    default=DEFAULT_PRIMARY_COLLECTION,
    show_default=True,
    help="Primary STAC collection to search (L1C)",
)
@click.option(
    "--collection",
    required=False,
    default=None,
    help="Backward-compatible alias for --collection1; takes precedence when set",
)
@click.option(
    "--collection2",
    required=False,
    default=DEFAULT_SECONDARY_COLLECTION,
    show_default=True,
    help="Secondary STAC collection used for L2A pairing",
)
@click.option(
    "--catalog_url",
    required=False,
    default=None,
    help="STAC catalog URL (defaults to CATALOG_URL env var)",
)
@click.option(
    "--cloud_cover",
    callback=parse_optional_float,
    default=str(DEFAULT_CLOUD_COVER),
    show_default=True,
    help="Max cloud cover (eo:cloud_cover<=value). Requires STAC Query extension.",
)
@click.option(
    "--limit",
    callback=parse_optional_int,
    default=str(DEFAULT_LIMIT),
    show_default=True,
    help="Max number of L1C items to return.",
)
def main(
    bbox: list[float],
    start_datetime: str,
    end_datetime: str,
    collection1: str,
    collection: str | None,
    collection2: str,
    catalog_url: str | None,
    cloud_cover: float | None,
    limit: int | None,
):
    """
    Search STAC catalog and output item IDs as JSON array to stdout.

    This is designed to be used as the first step in an Argo workflow,
    with the output used for fan-out parallel processing.
    """
    # --collection is kept as a backward-compatible alias for --collection1
    primary_collection = collection or collection1

    if catalog_url is None:
        catalog_url = os.getenv("CATALOG_URL", "")
        if not catalog_url:
            logger.error("No catalog URL provided and CATALOG_URL env var not set")
            sys.exit(1)

    try:
        l1c_items = search_stac(
            bbox=bbox,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            collection=primary_collection,
            catalog_url=catalog_url,
            cloud_cover=cloud_cover,
            limit=limit,
        )

        l1c_items = deduplicate_items(l1c_items)

        l2a_limit = get_l2a_pairing_limit(limit)
        l2a_items = search_stac(
            bbox=bbox,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            collection=collection2,
            catalog_url=catalog_url,
            cloud_cover=cloud_cover,
            limit=l2a_limit,
        )

        l2a_items = deduplicate_items(l2a_items)

        output_obj = build_output_payload(l1c_items, l2a_items)
        # Output JSON array to stdout for Argo withParam
        print(json.dumps(output_obj))

    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


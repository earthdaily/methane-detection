"""
Process Single STAC Item for Methane Detection

This script is designed to run as a parallel step in an Argo workflow.
It receives a single STAC item ID and processes it for methane detection,
outputting results to the file system for artifact collection.
"""
import copy
import json
import logging
import math
import os
import sys
from typing import Any, Optional

import affine
import boto3
import click
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.mask
import rasterio.warp
from matplotlib.colors import LinearSegmentedColormap
from pystac_client import Client as PyStacClient
from rasterio.session import AWSSession
from scipy.ndimage import median_filter
from shapely.geometry import box

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Methane spectral templates for different Sentinel-2 platforms
TEMPLATES = {
    "sentinel-2a": np.array([0.01222575, -0.01222575]),
    "sentinel-2b": np.array([0.00832454, -0.00832454]),
    "sentinel-2c": np.array([0.00958984, -0.00958984]),
}

# Output directories
OUT_DIR = "out"
STAC_ITEMS_OUT = os.path.join(OUT_DIR, "stac_items")
ASSETS_OUT = os.path.join(OUT_DIR, "assets")
CLEAR_MASK_CODES = [4, 5, 6, 7, 11] # Based on https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/

# Fixed target grid for OpenLayers multi-source compatibility (identical GeoTransform per run)
TARGET_RES = (0.00018, 0.00018)  # ~20 m at equator in EPSG:4326
TARGET_CRS = "EPSG:4326"
TARGET_BLOCK_SIZE = 256
METHANE_RANGE = (-2.0, 2.0)
AVERAGED_RANGE = (-0.15, 0.15)
IME_PERCENTILE = 95


def double_bbox(bbox: list[float]) -> list[float]:
    """Expand a bounding box around its center by a factor of two."""
    cp = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return [
        2 * bbox[0] - cp[0],
        2 * bbox[1] - cp[1],
        2 * bbox[2] - cp[0],
        2 * bbox[3] - cp[1],
    ]

def ensure_output_directories():
    """Create output directories if they don't exist."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STAC_ITEMS_OUT, exist_ok=True)
    os.makedirs(ASSETS_OUT, exist_ok=True)


def compute_target_grid(
    bbox: list[float],
) -> tuple[affine.Affine, int, int]:
    """
    Compute a snapped target grid from the AOI so all scenes in a run share the same
    resolution, origin, CRS, and raster dimensions.
    """
    aoi = double_bbox(bbox)
    minx, miny, maxx, maxy = aoi[0], aoi[1], aoi[2], aoi[3]
    res_x, res_y = TARGET_RES[0], TARGET_RES[1]
    minx_snap = math.floor(minx / res_x) * res_x
    maxx_snap = math.ceil(maxx / res_x) * res_x
    miny_snap = math.floor(miny / res_y) * res_y
    maxy_snap = math.ceil(maxy / res_y) * res_y
    width = int(round((maxx_snap - minx_snap) / res_x))
    height = int(round((maxy_snap - miny_snap) / res_y))
    transform = affine.Affine(res_x, 0, minx_snap, 0, -res_y, maxy_snap)
    return transform, width, height


def get_catalog_url() -> str:
    """Get the STAC catalog URL from the environment."""
    catalog_url = os.getenv("CATALOG_URL", "")
    if not catalog_url:
        logger.error("CATALOG_URL environment variable not set")
        sys.exit(1)
    logger.info(f"Catalog URL: {catalog_url}")
    return catalog_url


def save_figure(
    fig: plt.Figure,
    out_path: str,
    dpi: int = 150,
    bbox_inches: str | None = None,
) -> None:
    """Save and close a matplotlib figure."""
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def build_cloud_mask(scene_classification: np.ndarray) -> np.ndarray:
    """Create a clear-sky mask from Sentinel-2 scene classification values."""
    cloud_mask = np.zeros_like(scene_classification)
    for clear_code in CLEAR_MASK_CODES:
        cloud_mask[scene_classification == clear_code] = 1
    return cloud_mask


def create_ime_mask(mf: np.ndarray) -> np.ndarray:
    """Create the plume mask used for IME visualization."""
    mask = np.zeros_like(mf)
    mask[mf > np.percentile(mf, IME_PERCENTILE)] = 1
    return mask


def save_band_visualization(
    out_img: np.ndarray,
    download_bands_list: list[str],
    item_id: str,
) -> None:
    """Save the band comparison plot."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(out_img[0], cmap=plt.cm.gray)
    ax[0].set_title(f"Band {download_bands_list[0]}")
    ax[1].imshow(out_img[1], cmap=plt.cm.gray)
    ax[1].set_title(f"Band {download_bands_list[1]}")
    ax[2].imshow(out_img[1] - out_img[0], cmap=plt.cm.gray)
    ax[2].set_title("Band Difference")
    plt.tight_layout()
    save_figure(
        fig,
        os.path.join(ASSETS_OUT, f"{item_id}_bands_and_differences.png"),
    )
    logger.info("Band visualization saved")


def save_methane_visualization(
    data: np.ndarray,
    item_id: str,
    filename: str,
    title: str,
    colorbar_label: str,
    value_range: tuple[float, float],
) -> None:
    """Save a heatmap visualization for methane outputs."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, cmap=plt.cm.RdBu_r, vmin=value_range[0], vmax=value_range[1])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, os.path.join(ASSETS_OUT, filename))
    logger.info(f"Saved visualization: {filename}")


def apply_l2a_cloud_mask(
    bbox: list[float],
    l2a_item_id: str,
    catalog_url: str,
    aws_session: AWSSession,
    dst_tfm: affine.Affine,
    width: int,
    height: int,
    out_img: np.ndarray,
) -> np.ndarray:
    """Apply L2A-derived cloud mask and export RGB visualization."""
    l2a_results = read_and_reproject_data(
        bbox=bbox,
        collection="sentinel-2-l2a",
        item_id=l2a_item_id,
        catalog_url=catalog_url,
        download_bands_list=["scl", "visual"],
        aws_session=aws_session,
        transform_properties=(dst_tfm, width, height),
    )
    if l2a_results is None:
        logger.warning(f"Skipping L2A masking because item could not be read: {l2a_item_id}")
        return out_img

    l2a_img = l2a_results[0]
    scene_classification = l2a_img[0]
    rgb = l2a_img[1:]

    rgb_visual_path = os.path.join(ASSETS_OUT, f"{l2a_item_id}_rgb_visual.tif")
    create_ortho(rgb.astype(np.uint8), dst_tfm, rgb_visual_path)

    cloud_mask = build_cloud_mask(scene_classification)
    masked_img = out_img.copy()
    for band_idx in range(masked_img.shape[0]):
        masked_img[band_idx][cloud_mask != 1] = np.nan
    return masked_img


def write_time_signal_json(
    item_dict: dict[str, Any],
    values: list[dict[str, float | str]],
    item_id: str,
) -> None:
    """Write the per-item time-signal payload consumed by aggregation."""
    signal_json = {
        "datetime": item_dict["properties"]["datetime"],
        "values": values,
    }
    with open(os.path.join(ASSETS_OUT, f"{item_id}_time_signal.json"), "w") as f:
        json.dump(signal_json, f)


def process_regional_signal(
    bbox: list[float],
    ortho_path: str,
    dst_tfm: affine.Affine,
    item_dict: dict[str, Any],
    item_id: str,
    skip_viz: bool = False,
) -> None:
    """Generate averaged methane products and per-pixel time-signal JSON."""
    logger.info("Processing signal GeoJSON for regional averaging")

    polygon = box(*bbox)

    with rasterio.open(ortho_path) as src:
        inner_img, _ = rasterio.mask.mask(src, shapes=[polygon], crop=False, indexes=1)
        outer_img, _ = rasterio.mask.mask(
            src, shapes=[polygon], crop=False, invert=True, indexes=1
        )
        nodata = src.nodata
        averaged_mf = np.zeros_like(inner_img, dtype=np.float32)

    if nodata is None:
        inner_mask = np.isfinite(inner_img)
        outer_mask = np.isfinite(outer_img)
    else:
        inner_mask = inner_img != nodata
        outer_mask = outer_img != nodata

    if not np.any(inner_mask) or not np.any(outer_mask):
        raise ValueError("Regional averaging requires valid pixels both inside and outside the AOI")

    inner_signal = np.nanmean(inner_img[inner_mask])
    outer_signal = np.nanmean(outer_img[outer_mask])

    if np.any(inner_mask):
        averaged_mf[inner_mask] = inner_signal
    if np.any(outer_mask):
        averaged_mf[outer_mask] = outer_signal

    averaged_ortho_path = os.path.join(
        ASSETS_OUT, f"{item_id}_averaged_methane_enhancement.tif"
    )
    create_ortho(averaged_mf, dst_tfm, averaged_ortho_path)

    values = []
    for inner_data in inner_img[inner_mask]:
        adjusted_value = inner_data - outer_signal
        if not np.isnan(adjusted_value):
            values.append(
                {
                    "date": item_dict["properties"]["datetime"],
                    "value": float(adjusted_value),
                }
            )
    write_time_signal_json(item_dict, values, item_id)

    if not skip_viz:
        save_methane_visualization(
            averaged_mf,
            item_id=item_id,
            filename=f"{item_id}_averaged_methane_enhancement.png",
            title=f"Regional Averaged Methane - {item_id}",
            colorbar_label=r"Averaged methane column enhancement (mol m$^{-2}$)",
            value_range=AVERAGED_RANGE,
        )
        generate_colormap(
            vmin=AVERAGED_RANGE[0],
            vmax=AVERAGED_RANGE[1],
            fig_name=f"{item_id}_averaged_heatmap_colormap.png",
            colormap=plt.cm.RdBu_r,
        )


def read_and_reproject_data(
    bbox: list[float],
    collection: str,
    item_id: str,
    catalog_url: str,
    download_bands_list: list[str],
    aws_session: AWSSession,
    transform_properties: Optional[tuple[affine.Affine, int, int]] = None,
) -> Optional[tuple[np.ndarray, str, affine.Affine, int, int, dict[str, Any]]]:
    """
    Read STAC item and reproject bands to target bbox.

    Args:
        bbox: Bounding box [west, south, east, north]
        collection: STAC collection name
        item_id: STAC item ID to process
        catalog_url: STAC catalog URL
        download_bands_list: List of bands to download (e.g., ["B11", "B12"])
        aws_session: AWS session for S3 access

    Returns:
        Tuple of (reprojected_data, platform, transform, width, height, item_dict)
        or None if failed
    """
    logger.info(f"Searching for item: {item_id} in collection: {collection}")

    client = PyStacClient.open(catalog_url)
    results = client.search(ids=[item_id], collections=[collection])

    if results is None:
        logger.error("No results found for the given search")
        return None

    items = results.item_collection().items
    if len(items) == 0:
        logger.error(f"No item found with ID: {item_id}")
        return None

    item = items[0]
    logger.info(f"Found item: {item_id}")
    # Get platform information
    platform = item.properties.get("platform")
    if platform is None:
        logger.error("No platform found in item properties")
        return None

    logger.info(f"Platform: {platform}")

    # Get band HREFs
    try:
        band_hrefs = [item.assets[band].href for band in download_bands_list]
        logger.info(f"Bands to download: {download_bands_list}")
    except KeyError as e:
        logger.error(f"Band not found in item assets: {e}")
        return None

    # Save item metadata to file
    item_dict = item.to_dict()
    item_output_path = os.path.join(STAC_ITEMS_OUT, f"{item.id}.json")
    with open(item_output_path, "w") as f:
        json.dump(item_dict, f, indent=2)
    logger.info(f"Saved item metadata to: {item_output_path}")

    # Read and reproject the data
    logger.info(f"Reading and re-projecting data for platform {platform}")

    with rasterio.Env(aws_session):
        if transform_properties is not None:
            dst_tfm, width, height = transform_properties

        else:
            # Calculate target transform based on first band
            with rasterio.open(band_hrefs[0]) as src:
                dst_tfm, width, height = rasterio.warp.calculate_default_transform(
                    src.crs,
                    rasterio.crs.CRS.from_epsg(4326),
                    src.width,
                    src.height,
                    *src.bounds,
                )

                # Recalculate for target bbox with same resolution
                dst_tfm, width, height = rasterio.warp.calculate_default_transform(
                    rasterio.crs.CRS.from_epsg(4326),
                    rasterio.crs.CRS.from_epsg(4326),
                    width,
                    height,
                    *bbox,
                    resolution=(abs(dst_tfm[0]), abs(dst_tfm[4])),
                )

        logger.info(f"Target dimensions: {width}x{height}")

        # Prepare output array
        num_bands = len(download_bands_list)
        if 'visual' in download_bands_list: # visual is 3 bands so add more to get correct number of bands
            num_bands += 2

        out_img = np.empty((num_bands, height, width), dtype=np.float32)

        # Reproject each band
        im_ind = 0
        for ii, href in enumerate(band_hrefs):
            logger.info(f"Reprojecting band {download_bands_list[ii]} ({ii+1}/{len(band_hrefs)})")
            with rasterio.open(href) as src:
                for kk in range(src.count):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, kk+1),
                        destination=out_img[im_ind],
                        src_transform=src.transform,
                        dst_transform=dst_tfm,
                        src_crs=src.crs,
                        dst_crs=rasterio.crs.CRS.from_string(TARGET_CRS),
                        src_nodata=src.nodata,
                        dst_nodata=src.nodata,
                        resampling=rasterio.enums.Resampling.bilinear,
                    )
                    im_ind += 1

    logger.info("Data read and re-projected successfully")

    return out_img, platform, dst_tfm, width, height, item_dict


def scale_input_to_reference(input_map: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate scaling factor to align input map to reference.

    Args:
        input_map: Input array to scale
        reference: Reference array

    Returns:
        Scaling factor
    """
    valid_mask = ~np.isnan(input_map) * ~np.isnan(reference)
    reference_copy = reference[valid_mask]
    input_map_copy = input_map[valid_mask]
    scale = np.dot(reference_copy, input_map_copy) / np.dot(
        input_map_copy, input_map_copy
    )
    logger.debug(f"Calculated scale: {scale}")
    return scale


def normalize_inputs(
    data_or_signal: np.ndarray, normalization: float = 1.0
) -> np.ndarray:
    """
    Normalize inputs for matched filter.

    The normalization converts to appropriate units and removes the constant/monopole term.

    Args:
        data_or_signal: Multi-band array to normalize
        normalization: Initial normalization factor

    Returns:
        Normalized array
    """
    logger.info("Normalizing inputs")

    data_or_signal = data_or_signal * normalization

    if data_or_signal.ndim > 1:
        for ii in range(1, data_or_signal.shape[0]):
            scale = scale_input_to_reference(data_or_signal[ii], data_or_signal[0])
            data_or_signal[ii] *= scale

    return data_or_signal - np.nanmean(data_or_signal, axis=0)


def matched_filter(
    data: np.ndarray, cov: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """
    Apply matched filter for methane detection.

    Reference: https://faculty.nps.edu/rcristi/EO3404/C-Filters/text/3-Matched-Filters.pdf

    Args:
        data: N x H x W array of normalized methane band imagery
        cov: N x H x W array of per-pixel per-band variance
        template: N array of normalized signal to search for

    Returns:
        H x W array of matched filter response
    """
    logger.info("Applying matched filter")

    numerator = np.sum(data / cov * template[:, np.newaxis, np.newaxis], axis=0)
    denominator = np.sum(template[:, np.newaxis, np.newaxis] ** 2 / cov, axis=0)

    return numerator / denominator


def create_ortho(data: np.ndarray, dst_tfm: affine.Affine, out_path: str) -> str:
    """
    Create a single-resolution Cloud Optimized GeoTIFF (COG) from array data.

    Args:
        data: 2D or 3D array of data to write
        dst_tfm: Affine transform for georeferencing
        out_path: Output file path

    Returns:
        Path to created file
    """
    logger.info(f"Creating single-resolution COG at: {out_path}")

    if data.ndim == 2:
        height, width = data.shape
        count = 1
        write_data = data[np.newaxis, ...]

    elif data.ndim == 3:
        count, height, width = data.shape
        write_data = data

    else:
        raise ValueError("Input must be 2D or 3D")

    profile = copy.deepcopy(rasterio.default_gtiff_profile)
    profile.update({
        "driver": "COG",
        "width": width,
        "height": height,
        "count": count,
        "transform": dst_tfm,
        "nodata": 0,
        "dtype": write_data.dtype,
        "crs": rasterio.crs.CRS.from_string(TARGET_CRS),
        "blockxsize": TARGET_BLOCK_SIZE,
        "blockysize": TARGET_BLOCK_SIZE,
        "compress": "deflate",
        "overviews": "NONE",
    })

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(write_data)

    logger.info("COG created successfully")
    return out_path


def create_heatmap_cog(
    input_grayscale_path: str,
    output_rgb_path: str,
    mask: np.ndarray,
    input_min: float = -2.0,
    input_max: float = 2.0,
) -> str:
    """
    Create an RGB colorized single-resolution COG from grayscale GeoTIFF.

    Args:
        input_grayscale_path: Path to input grayscale GeoTIFF
        output_rgb_path: Path to output RGB GeoTIFF
        mask: Binary mask to apply (1 = data, 0 = nodata)
        input_min: Minimum value for color scale
        input_max: Maximum value for color scale

    Returns:
        Path to created file
    """
    logger.info(f"Creating single-resolution heatmap COG at: {output_rgb_path}")

    with rasterio.open(input_grayscale_path) as input_ds:
        profile = copy.deepcopy(input_ds.profile)
        data = input_ds.read(1)

        # Scale to 8-bit and apply colormap
        scaled = np.interp(data, (input_min, input_max), (0, 255)).astype(np.uint8)
        heatmap_arr = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)

    # Update profile for RGB output
    profile.update({
        "driver": "COG",
        "count": 3,
        "dtype": "uint8",
        "tiled": True,
        "interleave": "band",
        "nodata": 0,
        "BIGTIFF": "YES",
        "blockxsize": TARGET_BLOCK_SIZE,
        "blockysize": TARGET_BLOCK_SIZE,
        "compress": "deflate",
        "overviews": "NONE",
    })

    with rasterio.open(output_rgb_path, "w", **profile) as dst:
        for idx in range(3):
            # OpenCV uses BGR, so reverse the order and apply mask
            band_data = np.clip(heatmap_arr[:, :, 2 - idx], 1, 255) * mask
            dst.write(band_data, idx + 1)

    logger.info("Heatmap COG created successfully")
    return output_rgb_path


def generate_colormap(
    vmin: float, vmax: float, fig_name: str, colormap: LinearSegmentedColormap
) -> str:
    """
    Generate a standalone colormap legend image.

    Args:
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        fig_name: Output filename
        colormap: Matplotlib colormap to use

    Returns:
        Path to created file
    """
    output_path = os.path.join(ASSETS_OUT, fig_name)

    fig, ax = plt.subplots(figsize=(8, 1))
    dummy = np.empty(shape=(10, 10))
    dfig = plt.imshow(dummy, cmap=colormap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(dfig, ax=ax, location="bottom", orientation="horizontal")
    cbar.set_label(r"Methane column enhancement (mol m$^{-2}$)")
    ax.remove()
    save_figure(fig, output_path, bbox_inches="tight")

    logger.info(f"Colormap saved to: {output_path}")
    return output_path


def parse_list_string(ctx, param, value):
    """Parse JSON string to Python list (Click callback)."""
    try:
        return json.loads(value)
    except ValueError as e:
        logger.error(f"JSON parse error: {e}")
        raise click.BadParameter(f"Could not parse as JSON list: {value}")


@click.command()
@click.option(
    "--bbox",
    callback=parse_list_string,
    required=True,
    help="Bounding box as JSON list [west, south, east, north]",
)
@click.option(
    "--collection",
    required=True,
    help="STAC collection name",
)
@click.option(
    "--l1c-id",
    "l1c_item_id",
    required=True,
    help="L1C STAC item ID to process",
)
@click.option(
    "--l2a-id",
    "l2a_item_id",
    required=False,
    default=None,
    help="Optional L2A STAC item ID for cloud masking and RGB",
)
@click.option(
    "--download_bands_list",
    callback=parse_list_string,
    required=False,
    default='["B11.jp2", "B12.jp2"]',
    help="Bands to download as JSON list (default: [\"B11.jp2\", \"B12.jp2\"])",
)
@click.option(
    "--skip-viz/--no-viz",
    "skip_viz",
    is_flag=True,
    default=False,
    help="Skip matplotlib PNG debug outputs and legend generation.",
)
@click.option(
    "--skip-colorized/--no-colorized",
    "skip_colorized",
    is_flag=True,
    default=False,
    help="Skip creation of the colorized methane heatmap COG.",
)
@click.option(
    "--skip-overviews/--no-overviews",
    "skip_overviews",
    is_flag=True,
    default=False,
    help="Request single-resolution GeoTIFF outputs without overview generation.",
)
def main(
    bbox: list[float],
    collection: str,
    l1c_item_id: str,
    l2a_item_id: Optional[str] = None,
    download_bands_list: list[str] = ['B11.jp2', 'B12.jp2'],
    skip_viz: bool = False,
    skip_colorized: bool = False,
    skip_overviews: bool = False,
):
    """
    Process a single STAC item for methane detection.

    This script:
    1. Fetches the STAC item and downloads specified bands
    2. Reprojects data to the target bbox
    3. Applies matched filter for methane detection
    4. Generates output products (COGs, visualizations)
    5. Saves results to output directories for Argo artifact collection
    """
    logger.info(f"Starting methane processing for item: {l1c_item_id}")
    expanded_bbox = double_bbox(bbox)
    logger.info(f"Bbox: {expanded_bbox}")
    logger.info(f"Collection: {collection}")
    logger.info(f"Bands: {download_bands_list}")

    catalog_url = get_catalog_url()

    # Ensure output directories exist
    ensure_output_directories()

    try:
        target_transform, target_width, target_height = compute_target_grid(bbox)

        # Initialize AWS session
        aws_session = AWSSession(boto3.Session(), requester_pays=True)

        # Read and reproject data
        l1c_results = read_and_reproject_data(
            bbox=expanded_bbox,
            collection=collection,
            item_id=l1c_item_id,
            catalog_url=catalog_url,
            download_bands_list=download_bands_list,
            aws_session=aws_session,
            transform_properties=(target_transform, target_width, target_height),
        )

        if l1c_results is None:
            logger.error("Failed to read and reproject data")
            sys.exit(1)

        out_img, platform, dst_tfm, width, height, item_dict = l1c_results
        # Check if platform template exists
        if platform not in TEMPLATES:
            logger.error(f"No methane template found for platform: {platform}")
            sys.exit(1)

        if l2a_item_id is not None:
            out_img = apply_l2a_cloud_mask(
                bbox=expanded_bbox,
                l2a_item_id=l2a_item_id,
                catalog_url=catalog_url,
                aws_session=aws_session,
                dst_tfm=dst_tfm,
                width=width,
                height=height,
                out_img=out_img,
            )

        # Normalize inputs (convert from digital numbers to reflectance)
        out_img = normalize_inputs(out_img, 1 / 10000)

        # Create visualization of bands
        if skip_viz:
            logger.info("Skipping band comparison visualization (--no-viz)")
        else:
            save_band_visualization(out_img, download_bands_list, l1c_item_id)

        # Apply matched filter
        # Use identity covariance (assumes equal noise across bands)
        cov = np.ones_like(out_img)

        mf = matched_filter(out_img, cov, TEMPLATES[platform])
        mf = median_filter(mf, size=3)
        logger.info("Matched filter applied")

        if skip_viz:
            logger.info("Skipping methane enhancement visualization (--no-viz)")
        else:
            # Create main methane enhancement visualization
            save_methane_visualization(
                mf,
                item_id=l1c_item_id,
                filename=f"{l1c_item_id}_methane_enhancement.png",
                title=f"Methane Detection - {l1c_item_id}",
                colorbar_label=r"Methane column enhancement (mol m$^{-2}$)",
                value_range=METHANE_RANGE,
            )

            # Generate colormap legend
            generate_colormap(
                vmin=METHANE_RANGE[0],
                vmax=METHANE_RANGE[1],
                fig_name=f"{l1c_item_id}_heatmap_colormap.png",
                colormap=plt.cm.RdBu_r,
            )

        # Create IME mask (Integrated Mass Enhancement)
        # Mask pixels above 95th percentile as potential plumes
        mask = create_ime_mask(mf)
        logger.info(f"IME mask created ({IME_PERCENTILE}th percentile threshold)")

        if skip_viz:
            logger.info("Skipping masked methane visualization (--no-viz)")
        else:
            # Masked visualization
            save_methane_visualization(
                mf * mask,
                item_id=l1c_item_id,
                filename=f"{l1c_item_id}_methane_enhancement_masked.png",
                title=f"Methane Detection (IME Masked) - {l1c_item_id}",
                colorbar_label=r"Methane column enhancement (mol m$^{-2}$)",
                value_range=METHANE_RANGE,
            )

        # Create COG outputs (signal block uses ortho_path before warp, so run it first)
        ortho_path = create_ortho(
            mf,
            dst_tfm,
            os.path.join(ASSETS_OUT, f"{l1c_item_id}_methane_enhancement.tif"),
        )

        try:
            process_regional_signal(
                bbox=bbox,
                ortho_path=ortho_path,
                dst_tfm=dst_tfm,
                item_dict=item_dict,
                item_id=l1c_item_id,
                skip_viz=skip_viz,
            )
        except Exception as e:
            logger.warning(f"Failed to process signal GeoJSON: {e}")

        if skip_colorized:
            logger.info("Skipping colorized heatmap COG (--no-colorized)")
        else:
            colourized_cog_path = ortho_path.replace(".tif", "_colorized.tif")
            create_heatmap_cog(
                ortho_path,
                colourized_cog_path,
                mask,
                input_min=METHANE_RANGE[0],
                input_max=METHANE_RANGE[1],
            )

        logger.info(f"Processing completed successfully for item: {l1c_item_id}")
        logger.info(f"Outputs saved to: {ASSETS_OUT}")
        logger.info(f"STAC item metadata saved to: {STAC_ITEMS_OUT}")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


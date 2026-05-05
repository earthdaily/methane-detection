"""
Provider-specific configuration for STAC catalog access and S3 data retrieval.

Add a new entry to PROVIDERS to support a new backend — nothing else needs to change.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

__all__ = [
    "ProviderConfig",
    "PROVIDERS",
    "get_provider_config",
    "resolve_catalog_url",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration record for one STAC/S3 provider."""

    catalog_url: str
    """Default STAC catalog URL. Empty string means the operator MUST supply CATALOG_URL."""

    l1c_band_keys: list[str]
    """Default asset keys for L1C SWIR bands (used when --download-bands-list is omitted)."""

    l2a_band_keys: list[str]
    """Asset keys for L2A cloud-mask products: [scene_classification, rgb_visual]."""

    l2a_rgb_band_keys: tuple[str, ...]
    """
    Subset of l2a_band_keys that carry multi-band RGB visuals (and therefore
    bump the output band count by 2 in read_and_reproject_data).
    """

    s3_endpoint_url: Optional[str]
    """Non-AWS S3 endpoint URL (e.g. CDSE). None means standard AWS endpoint resolution."""

    s3_region: str
    """AWS region string passed to boto3 / AWSSession."""

    s3_virtual_hosting: bool
    """
    Controls the AWS_VIRTUAL_HOSTING rasterio env var.
    True  = virtual-host-style (standard AWS S3).
    False = path-style URLs (required for CDSE and other non-AWS S3).
    """

    requester_pays: bool
    """Whether to attach x-amz-request-payer=requester to S3 requests."""

    use_cql2_filter: bool
    """
    True  => use CQL2-JSON filter for cloud-cover queries (CDSE / OGC API Features).
    False => use the legacy STAC Query Extension (element84).
    """


PROVIDERS: dict[str, ProviderConfig] = {
    "e84": ProviderConfig(
        catalog_url="https://earth-search.aws.element84.com/v1",
        l1c_band_keys=["swir16", "swir22"],
        l2a_band_keys=["scl", "visual"],
        l2a_rgb_band_keys=("visual",),
        s3_endpoint_url=None,
        s3_region="us-west-2",
        s3_virtual_hosting=True,
        requester_pays=True,
        use_cql2_filter=False,
    ),
    "cdse": ProviderConfig(
        catalog_url="https://stac.dataspace.copernicus.eu/v1/",
        l1c_band_keys=["B11", "B12"],
        l2a_band_keys=["SCL_20m", "TCI_10m"],
        l2a_rgb_band_keys=("TCI_10m",),
        s3_endpoint_url="https://eodata.dataspace.copernicus.eu",
        s3_region="default",
        s3_virtual_hosting=False,
        requester_pays=False,
        use_cql2_filter=True,
    ),
    "ed": ProviderConfig(
        catalog_url="",  # no hardcoded default; CATALOG_URL env var is required
        l1c_band_keys=["B11.jp2", "B12.jp2"],
        l2a_band_keys=["SCL_20m", "TCI_10m"],
        l2a_rgb_band_keys=("TCI_10m",),
        s3_endpoint_url=None,
        s3_region="eu-central-1",
        s3_virtual_hosting=True,
        requester_pays=True,
        use_cql2_filter=False,
    ),
}


def get_provider_config(stac_provider: str) -> ProviderConfig:
    """
    Return the ProviderConfig for the given provider name.

    Args:
        stac_provider: Provider identifier (one of the keys of PROVIDERS).

    Returns:
        The matching ProviderConfig.

    Raises:
        ValueError: If stac_provider is not a registered provider name.
    """
    if stac_provider not in PROVIDERS:
        raise ValueError(
            f"Unknown STAC provider: {stac_provider!r}. "
            f"Valid choices: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[stac_provider]


def resolve_catalog_url(stac_provider: str) -> str:
    """
    Return the effective STAC catalog URL.

    Checks the CATALOG_URL env var first; falls back to the provider default.

    Raises:
        RuntimeError: If neither CATALOG_URL is set nor the provider has a
            hardcoded default (e.g. the "ed" provider). CLI entry points should
            catch this and exit with a friendly message.
    """
    catalog_url = os.getenv("CATALOG_URL", "")
    if catalog_url:
        logger.info(f"Catalog URL: {catalog_url}")
        return catalog_url

    default = get_provider_config(stac_provider).catalog_url
    if not default:
        raise RuntimeError(
            f"No catalog URL available for provider {stac_provider!r}: "
            "set the CATALOG_URL environment variable."
        )
    logger.info(
        f"No CATALOG_URL set; using {stac_provider} default: {default}"
    )
    return default

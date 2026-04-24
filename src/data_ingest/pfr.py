"""PFR advanced stats ingestion via direct nflverse parquet download.

nfl_data_py doesn't support PFR, so we download directly from GitHub releases.
"""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.data_ingest._nflverse_download import download_nflverse_parquet
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)

STAT_TYPES = ["pass", "rush", "rec", "def"]


def ingest_pfr(
    seasons: List[int] = None,
    stat_types: List[str] = None,
    force: bool = False,
) -> None:
    """Download PFR advanced stats parquet files from nflverse releases."""
    if seasons is None:
        seasons = list(range(config.PFR_MIN_SEASON, config.MAX_SEASON + 1))
    if stat_types is None:
        stat_types = STAT_TYPES

    for stat_type in stat_types:
        for season in seasons:
            dest = paths.RAW_PFR / f"pfr_{stat_type}_{season}.parquet"
            if dest.exists() and not force:
                continue

            logger.info(f"Ingesting PFR {stat_type} {season}...")
            filename = f"advstats_week_{stat_type}_{season}.parquet"
            result = download_nflverse_parquet(
                "pfr_advstats", filename, paths.RAW_PFR, force=force,
            )
            if result is None:
                logger.warning(f"PFR {stat_type} {season} not available")

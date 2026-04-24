"""FTN charting data ingestion — feature-flagged, 2022+ only.

FTN data lacks team columns; must be joined with PBP on play_id to get posteam/defteam.
"""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.data_ingest._nflverse_download import download_nflverse_parquet
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_ftn(
    seasons: List[int] = None,
    force: bool = False,
) -> None:
    """Download FTN charting data from nflverse releases.

    Only runs if FTN_ENABLED is True in config.
    """
    if not config.FTN_ENABLED:
        logger.info("FTN ingestion disabled by feature flag")
        return

    if seasons is None:
        seasons = list(range(config.FTN_MIN_SEASON, config.MAX_SEASON + 1))

    for season in seasons:
        dest = paths.RAW_FTN / f"ftn_{season}.parquet"
        if dest.exists() and not force:
            continue

        logger.info(f"Ingesting FTN charting {season}...")
        filename = f"ftn_charting_{season}.parquet"
        result = download_nflverse_parquet(
            "ftn_charting", filename, paths.RAW_FTN, force=force,
        )
        if result is None:
            logger.warning(f"FTN {season} not available")

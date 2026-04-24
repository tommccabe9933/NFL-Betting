"""Injury report ingestion from nflverse."""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_injuries(
    seasons: List[int] = None,
    force: bool = False,
) -> None:
    """Download and persist injury data by season."""
    if seasons is None:
        seasons = list(range(config.MIN_SEASON, config.MAX_SEASON + 1))

    for season in seasons:
        dest = paths.RAW_INJURIES / f"injuries_{season}.parquet"
        if dest.exists() and not force:
            continue

        logger.info(f"Ingesting injuries {season}...")
        try:
            import nfl_data_py as nfl
            pdf = nfl.import_injuries([season])
            df = pl.from_pandas(pdf)
            write_parquet(df, dest)
        except Exception as e:
            logger.warning(f"Failed injuries {season}: {e}")

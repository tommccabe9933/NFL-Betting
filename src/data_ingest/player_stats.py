"""Weekly player stats ingestion from nflverse."""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_player_stats(
    seasons: List[int] = None,
    force: bool = False,
) -> None:
    """Download and persist weekly player stats by season."""
    if seasons is None:
        seasons = list(range(config.MIN_SEASON, config.MAX_SEASON + 1))

    for season in seasons:
        dest = paths.RAW_PLAYER_STATS / f"player_stats_{season}.parquet"
        if dest.exists() and not force:
            continue

        logger.info(f"Ingesting player stats {season}...")
        try:
            import nfl_data_py as nfl
            pdf = nfl.import_weekly_data([season])
            df = pl.from_pandas(pdf)
            write_parquet(df, dest)
        except Exception as e:
            logger.warning(f"Failed player stats {season}: {e}")

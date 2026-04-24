"""Schedule data ingestion from nflverse."""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_schedules(
    seasons: List[int] = None,
    force: bool = False,
) -> None:
    """Download and persist schedule data.

    nfl_data_py returns all seasons in one call, so we save per-season.
    """
    if seasons is None:
        seasons = list(range(config.MIN_SEASON, config.MAX_SEASON + 1))

    # Check if we need to refresh
    if not force and all(
        (paths.RAW_SCHEDULES / f"schedules_{s}.parquet").exists()
        for s in seasons
    ):
        logger.debug("All schedule files exist, skipping")
        return

    logger.info(f"Ingesting schedules for {seasons[0]}-{seasons[-1]}...")
    try:
        import nfl_data_py as nfl
        pdf = nfl.import_schedules(seasons)
        df = pl.from_pandas(pdf)
    except Exception as e:
        logger.error(f"Failed to import schedules: {e}")
        return

    # Save per-season
    for season in seasons:
        season_df = df.filter(pl.col("season") == season)
        if len(season_df) > 0:
            write_parquet(
                season_df,
                paths.RAW_SCHEDULES / f"schedules_{season}.parquet",
            )

"""Next Gen Stats ingestion from nflverse."""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)

STAT_TYPES = ["passing", "rushing", "receiving"]


def ingest_ngs(
    seasons: List[int] = None,
    stat_types: List[str] = None,
    force: bool = False,
) -> None:
    """Download and persist NGS data by stat type and season."""
    if seasons is None:
        seasons = list(range(config.NGS_MIN_SEASON, config.MAX_SEASON + 1))
    if stat_types is None:
        stat_types = STAT_TYPES

    for stat_type in stat_types:
        for season in seasons:
            dest = paths.RAW_NGS / f"ngs_{stat_type}_{season}.parquet"
            if dest.exists() and not force:
                continue

            logger.info(f"Ingesting NGS {stat_type} {season}...")
            try:
                import nfl_data_py as nfl
                pdf = nfl.import_ngs_data(stat_type, [season])
                df = pl.from_pandas(pdf)
                write_parquet(df, dest)
            except Exception as e:
                logger.warning(f"Failed NGS {stat_type} {season}: {e}")

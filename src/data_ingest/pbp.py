"""Play-by-play data ingestion from nflverse."""

import logging
from typing import List

import polars as pl

from src import config, paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_pbp(
    seasons: List[int] = None,
    force: bool = False,
) -> None:
    """Download and persist PBP data by season.

    Uses nfl_data_py when available, falls back to direct download.
    Keeps only columns needed for feature engineering (~20 of ~400).
    """
    if seasons is None:
        seasons = list(range(config.MIN_SEASON, config.MAX_SEASON + 1))

    for season in seasons:
        dest = paths.RAW_PBP / f"pbp_{season}.parquet"
        if dest.exists() and not force:
            logger.debug(f"PBP {season} exists, skipping")
            continue

        logger.info(f"Ingesting PBP {season}...")
        df = _fetch_pbp(season)
        if df is not None and len(df) > 0:
            # Keep only needed columns (drop ~380 unneeded)
            keep = [c for c in config.PBP_KEEP_COLS if c in df.columns]
            df = df.select(keep)
            write_parquet(df, dest)
        else:
            logger.warning(f"No PBP data for {season}")


def _fetch_pbp(season: int) -> pl.DataFrame:
    """Fetch PBP for a single season."""
    try:
        import nfl_data_py as nfl
        pdf = nfl.import_pbp_data([season])
        return pl.from_pandas(pdf)
    except Exception as e:
        logger.warning(f"nfl_data_py failed for PBP {season}: {e}")
        # Fall back to direct download
        from src.data_ingest._nflverse_download import download_nflverse_parquet
        path = download_nflverse_parquet(
            "pbp", f"play_by_play_{season}.parquet", paths.RAW_PBP
        )
        if path:
            return pl.read_parquet(path)
        return None

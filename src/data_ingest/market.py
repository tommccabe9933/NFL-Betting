"""Market data ingestion — converts existing Kaggle CSVs to parquet."""

import logging

import polars as pl

from src import paths
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def ingest_market(force: bool = False) -> None:
    """Convert Spreadspoke game CSV to parquet.

    Source: data/nfl_all_seasons.csv (Kaggle Spreadspoke dataset)
    Contains: spread, total, moneyline, weather, stadium, neutral flag
    """
    dest = paths.RAW_MARKET / "spreadspoke_games.parquet"
    if dest.exists() and not force:
        logger.debug("Spreadspoke parquet exists, skipping")
        return

    csv_path = paths.LEGACY_DATA / "nfl_all_seasons.csv"
    if not csv_path.exists():
        logger.error(f"Spreadspoke CSV not found: {csv_path}")
        return

    logger.info("Converting Spreadspoke CSV to parquet...")
    df = pl.read_csv(
        csv_path,
        infer_schema_length=10000,
        ignore_errors=True,
        schema_overrides={
            "weather_temperature": pl.Utf8,
            "weather_wind_mph": pl.Utf8,
            "weather_humidity": pl.Utf8,
        },
    )
    # Normalize column names
    df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})

    # Current-season rows in Spreadspoke format (scripts/02_prepare_recent_data.R)
    for extra in sorted(paths.LEGACY_DATA.glob("nfl_games_20*.csv")):
        cur = pl.read_csv(extra, infer_schema_length=10000, schema_overrides={
            "weather_temperature": pl.Utf8, "weather_wind_mph": pl.Utf8,
            "weather_humidity": pl.Utf8, "schedule_week": pl.Utf8})
        df = pl.concat([df.with_columns(pl.col("schedule_week").cast(pl.Utf8)), cur], how="diagonal_relaxed")
    # Cast weather columns to numeric (non-numeric values become null)
    for col in ["weather_temperature", "weather_wind_mph", "weather_humidity"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    write_parquet(df, dest)

    # Also handle 2025 if present
    csv_2025 = paths.LEGACY_DATA / "nfl_2025.csv"
    if csv_2025.exists():
        dest_2025 = paths.RAW_MARKET / "nfl_2025.parquet"
        if not dest_2025.exists() or force:
            df_2025 = pl.read_csv(csv_2025, infer_schema_length=10000)
            df_2025 = df_2025.rename(
                {c: c.lower().replace(" ", "_") for c in df_2025.columns}
            )
            write_parquet(df_2025, dest_2025)

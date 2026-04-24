"""EPA-based team ratings from play-by-play data.

Computes rolling lagged team efficiency metrics. This is the core analytical
feature family — EPA per play, success rate, pass/rush splits, CPOE.

Replicates the R pipeline's EPA computation (scripts/01_build_model_data.R:30-60).
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def _aggregate_weekly_epa() -> pl.DataFrame:
    """Aggregate PBP to team-week EPA metrics.

    Filters: REG season, rush or pass plays, non-null EPA.
    """
    pbp = read_parquet_dir(paths.RAW_PBP)

    # Filter to valid plays
    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & ((pl.col("rush") == 1) | (pl.col("pass") == 1))
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & (pl.col("posteam") != "")
    )

    # Offensive EPA aggregation
    off_epa = (
        plays.group_by(["season", "week", "posteam"])
        .agg([
            pl.col("epa").mean().alias("off_epa_play"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_rush_epa"),
            (pl.col("epa") > 0).mean().alias("off_success_rate"),
            pl.col("cpoe").mean().alias("off_cpoe"),
        ])
        .rename({"posteam": "team"})
    )

    # Defensive EPA aggregation
    def_epa = (
        plays.group_by(["season", "week", "defteam"])
        .agg([
            pl.col("epa").mean().alias("def_epa_play"),
            (pl.col("epa") > 0).mean().alias("def_success_rate"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_rush_epa"),
        ])
        .rename({"defteam": "team"})
    )

    # Join offense and defense
    weekly = off_epa.join(def_epa, on=["season", "week", "team"], how="left")

    # Normalize team names
    weekly = weekly.with_columns(normalize_team_expr("team"))

    logger.info(f"Aggregated weekly EPA: {len(weekly)} team-weeks")
    return weekly


def build_epa_ratings() -> pl.DataFrame:
    """Build EPA-based rolling team ratings.

    Output: one row per (season, week, team) with lagged rolling features.
    Rolling windows span season boundaries (intentional — matches R behavior).
    """
    weekly = _aggregate_weekly_epa()

    # Columns to roll
    epa_cols = [
        "off_epa_play", "off_pass_epa", "off_rush_epa",
        "off_success_rate", "off_cpoe",
        "def_epa_play", "def_success_rate", "def_pass_epa", "def_rush_epa",
    ]

    # 5-game rolling means (lagged by 1)
    result = lagged_rolling_mean_multi(
        weekly, epa_cols,
        window=config.ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll5",
    )

    # 3-game rolling means (lagged by 1)
    result = lagged_rolling_mean_multi(
        result, epa_cols,
        window=config.SHORT_ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll3",
    )

    write_parquet(result, paths.FEATURE_STORE / "epa_ratings.parquet")
    logger.info(f"Built EPA ratings: {len(result)} team-weeks")
    return result

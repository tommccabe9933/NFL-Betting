"""Kaggle team stats features — operational metrics from existing CSVs.

These provide base features available 2012+ from the Kaggle dataset,
separate from the PBP-derived EPA features.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def build_kaggle_team_features() -> pl.DataFrame:
    """Build operational team features from Kaggle CSVs.

    Features: yards_per_play, pass_eff, rush_eff, comp_rate,
    third_down_rate, turnover_margin, sack_rate, pts_per_play.
    All rolled 5-game and 3-game lagged.
    """
    off_path = paths.LEGACY_DATA / "weekly_team_stats_offense.csv"
    def_path = paths.LEGACY_DATA / "weekly_team_stats_defense.csv"

    if not off_path.exists():
        logger.warning(f"Kaggle offense CSV not found: {off_path}")
        return pl.DataFrame()

    off = pl.read_csv(off_path, infer_schema_length=10000)
    off = off.rename({c: c.lower().replace(" ", "_") for c in off.columns})

    # Filter to regular season
    if "season_type" in off.columns:
        off = off.filter(pl.col("season_type") == "REG")

    # Normalize team
    if "team" in off.columns:
        off = off.with_columns(normalize_team_expr("team"))

    # Compute per-play metrics
    derived = off.with_columns([
        # Yards per play
        (pl.col("total_off_yards") / pl.col("off_plays")).alias("yards_per_play"),
        # Pass efficiency
        (pl.col("passing_yards") / pl.col("pass_attempts")).alias("pass_eff"),
        # Rush efficiency
        (pl.col("rushing_yards") / pl.col("rushing_attempts")).alias("rush_eff"),
        # Points per play
        (pl.col("points") / pl.col("off_plays")).alias("pts_per_play"),
    ]) if all(c in off.columns for c in ["total_off_yards", "off_plays"]) else off

    # Select available metrics
    metric_cols = [
        c for c in [
            "yards_per_play", "pass_eff", "rush_eff", "pts_per_play",
        ] if c in derived.columns
    ]

    if not metric_cols:
        logger.warning("No computable metrics from Kaggle offense CSV")
        return pl.DataFrame()

    weekly = derived.select(["season", "week", "team"] + metric_cols)

    # 5-game rolling
    result = lagged_rolling_mean_multi(
        weekly, metric_cols,
        window=config.ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll5",
    )

    # 3-game rolling
    result = lagged_rolling_mean_multi(
        result, metric_cols,
        window=config.SHORT_ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll3",
    )

    write_parquet(result, paths.FEATURE_STORE / "kaggle_team_stats.parquet")
    logger.info(f"Built Kaggle team stats: {len(result)} team-weeks")
    return result

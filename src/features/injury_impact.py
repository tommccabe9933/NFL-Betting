"""Injury impact features — positional-weighted injury scores.

Replicates R pipeline's injury impact scoring (scripts/10_build_advanced_features.R).
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def build_injury_features() -> pl.DataFrame:
    """Build injury impact features.

    For each team-week, computes:
    - injury_impact_score: sum of (position_weight * status_weight) for all injured players
    - n_out: count of players listed Out
    - n_questionable: count of players listed Questionable
    - qb_out: whether a QB is listed Out
    """
    try:
        injuries = read_parquet_dir(paths.RAW_INJURIES)
    except FileNotFoundError:
        logger.warning("No injury files found")
        return pl.DataFrame()

    # Map position and status to weights
    injuries = injuries.with_columns([
        pl.col("position")
        .map_elements(
            lambda x: config.INJURY_POS_WEIGHTS.get(x, 1),
            return_dtype=pl.Int64,
        )
        .alias("pos_weight"),
        pl.col("report_status")
        .map_elements(
            lambda x: config.INJURY_STATUS_WEIGHTS.get(x, 0.0),
            return_dtype=pl.Float64,
        )
        .alias("status_weight"),
    ])

    injuries = injuries.with_columns(
        (pl.col("pos_weight") * pl.col("status_weight")).alias("impact")
    )

    # Aggregate per team-week
    team_injury = (
        injuries.group_by(["season", "week", "team"])
        .agg([
            pl.col("impact").sum().alias("injury_impact_score"),
            (pl.col("report_status") == "Out").sum().alias("n_out"),
            (pl.col("report_status") == "Questionable").sum().alias("n_questionable"),
            (
                (pl.col("position") == "QB") & (pl.col("report_status") == "Out")
            ).any().alias("qb_out"),
        ])
    )

    team_injury = team_injury.with_columns(normalize_team_expr("team"))

    write_parquet(team_injury, paths.FEATURE_STORE / "injury_impact.parquet")
    logger.info(f"Built injury impact: {len(team_injury)} team-weeks")
    return team_injury

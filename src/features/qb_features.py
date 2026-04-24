"""QB features — rolling stats + uncertainty from depth charts and injuries.

Combines player_stats (QB weekly performance), depth_charts (who is QB1),
and injuries (is QB1 available) into pregame QB features.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def _get_qb_weekly_stats() -> pl.DataFrame:
    """Extract QB-level weekly stats from player_stats."""
    try:
        stats = read_parquet_dir(paths.RAW_PLAYER_STATS)
    except FileNotFoundError:
        logger.warning("No player stats files found")
        return pl.DataFrame()

    # Filter to QBs with meaningful attempts
    qbs = stats.filter(
        (pl.col("position") == "QB")
        & (pl.col("attempts") >= 5)
    )

    # Aggregate to team-week (primary QB)
    # Take the QB with most attempts per team-week
    team_qb = (
        qbs.sort("attempts", descending=True)
        .group_by(["season", "week", "recent_team"])
        .first()
        .rename({"recent_team": "team"})
    )

    # Select relevant columns
    qb_cols = [
        "season", "week", "team", "player_name",
    ]
    # Add available stat columns
    stat_cols = [
        "passing_epa", "dakota", "attempts", "sacks",
        "passing_air_yards", "passing_yards_after_catch",
        "completions",
    ]
    available = [c for c in stat_cols if c in team_qb.columns]

    result = team_qb.select(qb_cols + available)

    # Compute derived stats
    if "passing_epa" in result.columns and "attempts" in result.columns:
        result = result.with_columns(
            (pl.col("passing_epa") / pl.col("attempts")).alias("qb_epa_per_play")
        )
    if "sacks" in result.columns and "attempts" in result.columns:
        result = result.with_columns(
            (pl.col("sacks") / (pl.col("attempts") + pl.col("sacks"))).alias("qb_sack_rate")
        )
    if "passing_air_yards" in result.columns and "attempts" in result.columns:
        result = result.with_columns(
            (pl.col("passing_air_yards") / pl.col("attempts")).alias("qb_air_yards_per_att")
        )

    # CPOE: use dakota if available (it's nflverse's CPOE metric)
    if "dakota" in result.columns:
        result = result.with_columns(pl.col("dakota").alias("qb_cpoe"))

    # Normalize teams
    result = result.with_columns(normalize_team_expr("team"))

    return result


def _get_qb_uncertainty() -> pl.DataFrame:
    """Determine QB availability from injuries and depth charts.

    Returns one row per (season, week, team) with:
    - qb_out: whether QB1 is listed as Out
    - qb_questionable: whether QB1 is listed as Questionable/Doubtful
    - qb_change_flag: whether QB1 changed from previous week
    """
    # Try loading injuries
    try:
        injuries = read_parquet_dir(paths.RAW_INJURIES)
    except FileNotFoundError:
        logger.warning("No injury files found for QB uncertainty")
        return pl.DataFrame()

    # Filter to QBs
    qb_injuries = injuries.filter(
        pl.col("position").is_in(["QB"])
    )

    if len(qb_injuries) == 0:
        return pl.DataFrame()

    # Get most severe status per team-week
    status_priority = {"Out": 4, "Doubtful": 3, "Questionable": 2, "Probable": 1}

    qb_status = (
        qb_injuries
        .with_columns(
            pl.col("report_status")
            .map_elements(lambda x: status_priority.get(x, 0), return_dtype=pl.Int64)
            .alias("_severity")
        )
        .sort("_severity", descending=True)
        .group_by(["season", "week", "team"])
        .first()
        .select([
            "season", "week", "team",
            (pl.col("report_status") == "Out").alias("qb_out"),
            pl.col("report_status").is_in(["Questionable", "Doubtful"]).alias("qb_questionable"),
        ])
    )

    qb_status = qb_status.with_columns([
        normalize_team_expr("team"),
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ])
    return qb_status


def build_qb_features() -> pl.DataFrame:
    """Build QB features: rolling performance + uncertainty."""
    qb_stats = _get_qb_weekly_stats()

    if len(qb_stats) == 0:
        logger.warning("No QB stats available — returning empty")
        return pl.DataFrame()

    # Rolling features
    roll_cols = [c for c in [
        "qb_epa_per_play", "qb_cpoe", "qb_sack_rate", "qb_air_yards_per_att"
    ] if c in qb_stats.columns]

    if roll_cols:
        result = lagged_rolling_mean_multi(
            qb_stats, roll_cols,
            window=config.ROLLING_WINDOW,
            group_col="team",
            sort_cols=["season", "week"],
            suffix="_roll5",
        )
    else:
        result = qb_stats

    # QB change flag (different starter from previous week)
    if "player_name" in result.columns:
        result = result.sort(["season", "week"]).with_columns(
            (pl.col("player_name") != pl.col("player_name").shift(1).over("team"))
            .alias("qb_change_flag")
        )

    # Merge uncertainty
    uncertainty = _get_qb_uncertainty()
    if len(uncertainty) > 0:
        result = result.join(
            uncertainty, on=["season", "week", "team"], how="left"
        ).with_columns([
            pl.col("qb_out").fill_null(False),
            pl.col("qb_questionable").fill_null(False),
        ])

    write_parquet(result, paths.FEATURE_STORE / "qb_features.parquet")
    logger.info(f"Built QB features: {len(result)} team-weeks")
    return result

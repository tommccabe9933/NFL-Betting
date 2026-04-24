"""Tracking proxy features from Next Gen Stats.

NGS provides public weekly player-level metrics that serve as proxies for
tracking data: time to throw, aggressiveness, rush yards over expected, etc.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def _aggregate_ngs_passing() -> pl.DataFrame:
    """Aggregate NGS passing stats to team-week."""
    try:
        ngs_files = sorted(paths.RAW_NGS.glob("ngs_passing_*.parquet"))
        if not ngs_files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(f) for f in ngs_files]
        ngs = pl.concat(dfs)
    except Exception as e:
        logger.warning(f"Failed to load NGS passing: {e}")
        return pl.DataFrame()

    # Identify available columns
    pass_metrics = []
    col_map = {
        "avg_time_to_throw": "ngs_time_to_throw",
        "avg_completed_air_yards": "ngs_completed_air_yards",
        "aggressiveness": "ngs_aggressiveness",
        "completion_percentage_above_expectation": "ngs_cpoe",
        "avg_air_yards_to_sticks": "ngs_air_yards_to_sticks",
    }

    for orig, alias in col_map.items():
        if orig in ngs.columns:
            pass_metrics.append((orig, alias))

    if not pass_metrics:
        return pl.DataFrame()

    # Aggregate to team-week (weighted by attempts if available)
    agg_exprs = [pl.col(orig).mean().alias(alias) for orig, alias in pass_metrics]
    team_col = "team_abbr" if "team_abbr" in ngs.columns else "team"

    team_ngs = (
        ngs.group_by(["season", "week", team_col])
        .agg(agg_exprs)
        .rename({team_col: "team"})
    )

    return team_ngs


def _aggregate_ngs_rushing() -> pl.DataFrame:
    """Aggregate NGS rushing stats to team-week."""
    try:
        ngs_files = sorted(paths.RAW_NGS.glob("ngs_rushing_*.parquet"))
        if not ngs_files:
            return pl.DataFrame()
        dfs = [pl.read_parquet(f) for f in ngs_files]
        ngs = pl.concat(dfs)
    except Exception:
        return pl.DataFrame()

    rush_metrics = []
    col_map = {
        "rush_yards_over_expected": "ngs_rush_yoe",
        "efficiency": "ngs_rush_efficiency",
        "rush_yards_over_expected_per_att": "ngs_rush_yoe_per_att",
        "percent_attempts_gte_eight_defenders": "ngs_rush_8plus_def",
    }

    for orig, alias in col_map.items():
        if orig in ngs.columns:
            rush_metrics.append((orig, alias))

    if not rush_metrics:
        return pl.DataFrame()

    agg_exprs = [pl.col(orig).mean().alias(alias) for orig, alias in rush_metrics]
    team_col = "team_abbr" if "team_abbr" in ngs.columns else "team"

    team_ngs = (
        ngs.group_by(["season", "week", team_col])
        .agg(agg_exprs)
        .rename({team_col: "team"})
    )

    return team_ngs


def build_tracking_proxy_features() -> pl.DataFrame:
    """Build tracking proxy features from NGS data with lagged rolling."""
    passing = _aggregate_ngs_passing()
    rushing = _aggregate_ngs_rushing()

    if len(passing) == 0 and len(rushing) == 0:
        logger.warning("No NGS data available")
        return pl.DataFrame()

    # Merge passing and rushing
    if len(passing) > 0 and len(rushing) > 0:
        weekly = passing.join(rushing, on=["season", "week", "team"], how="outer")
    elif len(passing) > 0:
        weekly = passing
    else:
        weekly = rushing

    weekly = weekly.with_columns(normalize_team_expr("team"))

    # Rolling features
    ngs_cols = [c for c in weekly.columns if c.startswith("ngs_")]
    if ngs_cols:
        result = lagged_rolling_mean_multi(
            weekly, ngs_cols,
            window=config.ROLLING_WINDOW,
            group_col="team",
            sort_cols=["season", "week"],
            suffix="_roll5",
        )
    else:
        result = weekly

    write_parquet(result, paths.FEATURE_STORE / "tracking_proxy.parquet")
    logger.info(f"Built tracking proxy: {len(result)} team-weeks")
    return result

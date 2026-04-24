"""PFR pressure/advanced stats features — 2018+ only.

Builds pressure_rate, drop_rate, bad_throw_rate from PFR advanced stats.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet

logger = logging.getLogger(__name__)


def _load_pfr(stat_type: str) -> pl.DataFrame:
    """Load PFR parquet files for a stat type."""
    files = sorted(paths.RAW_PFR.glob(f"pfr_{stat_type}_*.parquet"))
    if not files:
        files = sorted(paths.RAW_PFR.glob(f"advstats_week_{stat_type}_*.parquet"))
    if not files:
        files = sorted(paths.RAW_PFR.glob(f"advstats_season_{stat_type}_*.parquet"))
    if not files:
        return pl.DataFrame()
    dfs = [pl.read_parquet(f) for f in files]
    return pl.concat(dfs)


def build_pfr_features() -> pl.DataFrame:
    """Build PFR-based features.

    From pass stats: pressure_rate, drop_rate, bad_throw_rate (offensive)
    From def stats: def_pressure_rate, def_comp_allowed_rate
    All rolled 5-game lagged. Available 2018+ only.
    """
    pass_df = _load_pfr("pass")
    def_df = _load_pfr("def")

    if len(pass_df) == 0 and len(def_df) == 0:
        logger.warning("No PFR data available")
        return pl.DataFrame()

    frames = []

    # Offensive pass pressure stats
    if len(pass_df) > 0:
        team_col = "team" if "team" in pass_df.columns else "tm"
        off_cols = {}
        if "times_pressured" in pass_df.columns and "attempts" in pass_df.columns:
            off_cols["pfr_off_pressure_rate"] = (
                pl.col("times_pressured") / pl.col("attempts")
            )
        if "drops" in pass_df.columns and "attempts" in pass_df.columns:
            off_cols["pfr_drop_rate"] = pl.col("drops") / pl.col("attempts")
        if "bad_throws" in pass_df.columns and "attempts" in pass_df.columns:
            off_cols["pfr_bad_throw_rate"] = pl.col("bad_throws") / pl.col("attempts")

        if off_cols:
            off_agg = (
                pass_df.group_by(["season", "week", team_col])
                .agg([expr.mean().alias(name) for name, expr in off_cols.items()])
                .rename({team_col: "team"})
            )
            frames.append(off_agg)

    # Defensive stats
    if len(def_df) > 0:
        team_col = "team" if "team" in def_df.columns else "tm"
        def_cols = {}
        if "hurries" in def_df.columns:
            def_cols["pfr_def_hurries"] = pl.col("hurries").mean()
        if "missed_tackles" in def_df.columns:
            def_cols["pfr_missed_tackles"] = pl.col("missed_tackles").mean()

        if def_cols:
            def_agg = (
                def_df.group_by(["season", "week", team_col])
                .agg([expr.alias(name) for name, expr in def_cols.items()])
                .rename({team_col: "team"})
            )
            frames.append(def_agg)

    if not frames:
        return pl.DataFrame()

    # Merge all PFR stats
    weekly = frames[0]
    for f in frames[1:]:
        weekly = weekly.join(f, on=["season", "week", "team"], how="outer")

    weekly = weekly.with_columns(normalize_team_expr("team"))

    # Rolling features
    pfr_cols = [c for c in weekly.columns if c.startswith("pfr_")]
    result = lagged_rolling_mean_multi(
        weekly, pfr_cols,
        window=config.ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll5",
    )

    # Add coverage flag
    result = result.with_columns(pl.lit(True).alias("has_pfr"))

    write_parquet(result, paths.FEATURE_STORE / "pfr_pressure.parquet")
    logger.info(f"Built PFR features: {len(result)} team-weeks")
    return result

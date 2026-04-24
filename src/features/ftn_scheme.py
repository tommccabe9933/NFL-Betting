"""FTN charting features — play action, RPO, blitz rates. 2022+ only.

FTN data lacks team columns; must join with PBP on play_id to get posteam/defteam.
Feature-flagged via config.FTN_ENABLED.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def build_ftn_features() -> pl.DataFrame:
    """Build FTN charting features.

    Requires PBP join to get team identifiers (posteam/defteam).

    Offensive: play_action_rate, rpo_rate, screen_rate
    Defensive: avg_blitzers, avg_pass_rushers
    All rolled 5-game lagged. 2022+ only.
    """
    if not config.FTN_ENABLED:
        logger.info("FTN disabled by feature flag")
        return pl.DataFrame()

    # Load FTN data
    ftn_files = sorted(paths.RAW_FTN.glob("ftn_*.parquet"))
    if not ftn_files:
        logger.warning("No FTN data files found")
        return pl.DataFrame()

    ftn = pl.concat([pl.read_parquet(f) for f in ftn_files])

    # Load PBP for team resolution
    pbp = read_parquet_dir(
        paths.RAW_PBP,
        columns=["season", "week", "play_id", "posteam", "defteam", "game_id"],
    )

    # Join FTN with PBP on play identifiers
    # FTN typically has nflverse_game_id and nflverse_play_id
    join_cols = []
    if "nflverse_game_id" in ftn.columns and "game_id" in pbp.columns:
        ftn = ftn.rename({"nflverse_game_id": "game_id"})
        join_cols.append("game_id")
    if "nflverse_play_id" in ftn.columns and "play_id" in pbp.columns:
        ftn = ftn.rename({"nflverse_play_id": "play_id"})
        join_cols.append("play_id")

    if not join_cols:
        # Try season + week + play_id
        join_cols = ["season", "week", "play_id"]

    # Cast join columns to same types
    for c in join_cols:
        if c in ftn.columns and c in pbp.columns:
            ftn = ftn.with_columns(pl.col(c).cast(pl.Int64, strict=False))
            pbp = pbp.with_columns(pl.col(c).cast(pl.Int64, strict=False))

    # Ensure join columns exist
    common = [c for c in join_cols if c in ftn.columns and c in pbp.columns]
    if not common:
        logger.error("Cannot join FTN with PBP — no common columns")
        return pl.DataFrame()

    ftn_with_teams = ftn.join(
        pbp.select(common + ["posteam", "defteam"]).unique(),
        on=common,
        how="inner",
    )

    match_rate = len(ftn_with_teams) / max(len(ftn), 1)
    logger.info(f"FTN-PBP join match rate: {match_rate:.1%}")
    if match_rate < 0.5:
        logger.warning("FTN-PBP match rate below 50% — check join columns")

    # Offensive aggregation
    off_agg_exprs = []
    if "is_play_action" in ftn_with_teams.columns:
        off_agg_exprs.append(pl.col("is_play_action").mean().alias("ftn_play_action_rate"))
    if "is_rpo" in ftn_with_teams.columns:
        off_agg_exprs.append(pl.col("is_rpo").mean().alias("ftn_rpo_rate"))
    if "is_screen_pass" in ftn_with_teams.columns:
        off_agg_exprs.append(pl.col("is_screen_pass").mean().alias("ftn_screen_rate"))

    # Defensive aggregation
    def_agg_exprs = []
    if "n_blitzers" in ftn_with_teams.columns:
        def_agg_exprs.append(pl.col("n_blitzers").mean().alias("ftn_avg_blitzers"))
    if "n_pass_rushers" in ftn_with_teams.columns:
        def_agg_exprs.append(pl.col("n_pass_rushers").mean().alias("ftn_avg_pass_rushers"))

    frames = []

    if off_agg_exprs:
        off = (
            ftn_with_teams.group_by(["season", "week", "posteam"])
            .agg(off_agg_exprs)
            .rename({"posteam": "team"})
        )
        frames.append(off)

    if def_agg_exprs:
        def_ = (
            ftn_with_teams.group_by(["season", "week", "defteam"])
            .agg(def_agg_exprs)
            .rename({"defteam": "team"})
        )
        frames.append(def_)

    if not frames:
        return pl.DataFrame()

    weekly = frames[0]
    for f in frames[1:]:
        weekly = weekly.join(f, on=["season", "week", "team"], how="outer")

    weekly = weekly.with_columns(normalize_team_expr("team"))

    # Rolling
    ftn_cols = [c for c in weekly.columns if c.startswith("ftn_")]
    result = lagged_rolling_mean_multi(
        weekly, ftn_cols,
        window=config.ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll5",
    )

    result = result.with_columns(pl.lit(True).alias("has_ftn"))

    write_parquet(result, paths.FEATURE_STORE / "ftn_scheme.parquet")
    logger.info(f"Built FTN features: {len(result)} team-weeks")
    return result

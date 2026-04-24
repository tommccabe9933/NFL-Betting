"""Situational EPA features — 3rd down, redzone, explosive plays.

Builds on PBP data with play-situation filters, then applies lagged rolling.
"""

import logging

import polars as pl

from src import config, paths
from src.normalize.teams import normalize_team_expr
from src.features._rolling import lagged_rolling_mean_multi
from src.utils.io import write_parquet, read_parquet_dir

logger = logging.getLogger(__name__)


def build_situational_epa() -> pl.DataFrame:
    """Build situational EPA features from PBP.

    Per team-week:
    - 3rd down EPA (offense/defense)
    - Redzone EPA (yardline_100 <= 20, offense/defense)
    - Early down EPA (1st/2nd, offense/defense)
    - Explosive rate (EPA > 1.5)
    - Negative play rate (EPA < -1)
    All rolled 5-game lagged.
    """
    pbp = read_parquet_dir(paths.RAW_PBP)

    plays = pbp.filter(
        (pl.col("season_type") == "REG")
        & ((pl.col("rush") == 1) | (pl.col("pass") == 1))
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & (pl.col("posteam") != "")
    )

    # Offensive situational stats
    off_sit = (
        plays.group_by(["season", "week", "posteam"])
        .agg([
            # 3rd down
            pl.col("epa").filter(pl.col("down") == 3).mean().alias("off_3rd_epa"),
            # Redzone
            pl.col("epa").filter(pl.col("yardline_100") <= 20).mean().alias("off_rz_epa"),
            # Early downs
            pl.col("epa").filter(pl.col("down").is_in([1, 2])).mean().alias("off_early_epa"),
            # Pass EPA
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("off_sit_pass_epa"),
            # Rush EPA
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("off_sit_rush_epa"),
            # Explosive rate
            (pl.col("epa") > 1.5).mean().alias("off_explosive_rate"),
            # Negative play rate
            (pl.col("epa") < -1.0).mean().alias("off_neg_play_rate"),
        ])
        .rename({"posteam": "team"})
    )

    # Defensive situational stats
    def_sit = (
        plays.group_by(["season", "week", "defteam"])
        .agg([
            pl.col("epa").filter(pl.col("down") == 3).mean().alias("def_3rd_epa"),
            pl.col("epa").filter(pl.col("yardline_100") <= 20).mean().alias("def_rz_epa"),
            pl.col("epa").filter(pl.col("down").is_in([1, 2])).mean().alias("def_early_epa"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("def_sit_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias("def_sit_rush_epa"),
            (pl.col("epa") > 1.5).mean().alias("def_explosive_rate"),
            (pl.col("epa") < -1.0).mean().alias("def_neg_play_rate"),
        ])
        .rename({"defteam": "team"})
    )

    weekly = off_sit.join(def_sit, on=["season", "week", "team"], how="left")
    weekly = weekly.with_columns(normalize_team_expr("team"))

    # Rolling features
    sit_cols = [c for c in weekly.columns if c.startswith(("off_", "def_")) and c != "team"]
    result = lagged_rolling_mean_multi(
        weekly, sit_cols,
        window=config.ROLLING_WINDOW,
        group_col="team",
        sort_cols=["season", "week"],
        suffix="_roll5",
    )

    write_parquet(result, paths.FEATURE_STORE / "situational_epa.parquet")
    logger.info(f"Built situational EPA: {len(result)} team-weeks")
    return result

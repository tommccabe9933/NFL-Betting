"""Rest and schedule features."""

import logging

import polars as pl

from src import paths
from src.utils.io import write_parquet, read_parquet

logger = logging.getLogger(__name__)


def build_rest_features() -> pl.DataFrame:
    """Build rest/schedule features from canonical games.

    Output: one row per (game_id_norm, team) with:
    - days_rest, short_week, is_primetime, div_game, is_dome, is_turf, neutral_site
    """
    games = read_parquet(paths.NORMALIZED / "canonical_games.parquet")
    games = games.filter(pl.col("game_type") == "REG")

    rows = []

    # Home team features
    home = games.select([
        "game_id_norm", "season", "week",
        pl.col("home_team").alias("team"),
        pl.col("home_rest").alias("days_rest"),
    ])

    # Away team features
    away = games.select([
        "game_id_norm", "season", "week",
        pl.col("away_team").alias("team"),
        pl.col("away_rest").alias("days_rest"),
    ])

    # Stack home and away
    rest = pl.concat([home, away])

    # Compute derived features
    rest = rest.with_columns([
        (pl.col("days_rest") <= 6).alias("short_week"),
        (pl.col("days_rest") >= 10).alias("mini_bye"),
    ])

    # Cap rest at 21 (bye weeks)
    rest = rest.with_columns(
        pl.col("days_rest").clip(upper_bound=21)
    )

    # Join game-level context features
    context = games.select([
        "game_id_norm",
        pl.col("div_game").cast(pl.Boolean).alias("div_game"),
        # Dome detection
        pl.col("roof").is_in(["dome", "closed"]).alias("is_dome"),
        # Surface
        (pl.col("surface") != "grass").alias("is_turf"),
        # Neutral site (may not exist in nflverse schedules)
        (pl.lit(False) if "stadium_neutral" not in games.columns
         else pl.when(pl.col("stadium_neutral").is_not_null())
         .then(pl.col("stadium_neutral").cast(pl.Boolean))
         .otherwise(pl.lit(False)))
        .alias("neutral_site"),
    ])

    result = rest.join(context, on="game_id_norm", how="left")

    write_parquet(result, paths.FEATURE_STORE / "rest_schedule.parquet")
    logger.info(f"Built rest/schedule: {len(result)} team-game entries")
    return result

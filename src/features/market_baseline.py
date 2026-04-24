"""Market baseline features — spread, total, moneyline, implied probabilities.

Treats the betting market as the baseline signal. All other features are overlays
around market error rather than replacements.
"""

import logging

import polars as pl

from src import paths
from src.utils.io import write_parquet, read_parquet

logger = logging.getLogger(__name__)


def _american_to_implied_prob(moneyline: pl.Expr) -> pl.Expr:
    """Convert American moneyline odds to implied probability."""
    return (
        pl.when(moneyline < 0)
        .then((-moneyline) / (-moneyline + 100))
        .when(moneyline > 0)
        .then(pl.lit(100.0) / (moneyline + 100))
        .otherwise(pl.lit(0.5))
    )


def build_market_features() -> pl.DataFrame:
    """Build market baseline features from canonical games.

    Output columns:
    - game_id_norm, season, week
    - fav_team, dog_team, fav_at_home
    - spread_fav (always positive — magnitude of spread)
    - over_under_line (total)
    - home_implied_prob, away_implied_prob (raw from moneylines)
    - fav_implied_prob_devigged, dog_implied_prob_devigged (de-vigged)
    """
    games = read_parquet(paths.NORMALIZED / "canonical_games.parquet")

    # Filter to regular season with valid spread
    games = games.filter(
        (pl.col("game_type") == "REG")
        & pl.col("spread_line").is_not_null()
    )

    # Compute implied probabilities from moneylines
    result = games.with_columns([
        # Raw implied probs
        _american_to_implied_prob(pl.col("home_moneyline")).alias("home_implied_prob"),
        _american_to_implied_prob(pl.col("away_moneyline")).alias("away_implied_prob"),
    ])

    # De-vig using multiplicative method: p_i / sum(p_i)
    result = result.with_columns([
        (pl.col("home_implied_prob") + pl.col("away_implied_prob")).alias("_total_implied"),
    ]).with_columns([
        (pl.col("home_implied_prob") / pl.col("_total_implied")).alias("home_devigged"),
        (pl.col("away_implied_prob") / pl.col("_total_implied")).alias("away_devigged"),
    ])

    # Assign fav/dog perspective
    result = result.with_columns([
        # Fav team
        pl.col("team_favorite_id").alias("fav_team"),
        # Dog team
        pl.when(pl.col("team_favorite_id") == pl.col("home_team"))
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))
        .alias("dog_team"),
        # Fav implied prob (de-vigged)
        pl.when(pl.col("team_favorite_id") == pl.col("home_team"))
        .then(pl.col("home_devigged"))
        .otherwise(pl.col("away_devigged"))
        .alias("fav_implied_prob_devigged"),
        # Dog implied prob (de-vigged)
        pl.when(pl.col("team_favorite_id") == pl.col("home_team"))
        .then(pl.col("away_devigged"))
        .otherwise(pl.col("home_devigged"))
        .alias("dog_implied_prob_devigged"),
    ])

    # Select output columns
    output = result.select([
        "game_id_norm", "season", "week",
        "fav_team", "dog_team",
        pl.col("fav_at_home").cast(pl.Boolean),
        pl.col("spread_favorite").alias("spread_fav"),
        pl.col("total_line").alias("over_under_line"),
        "home_moneyline", "away_moneyline",
        "home_implied_prob", "away_implied_prob",
        "fav_implied_prob_devigged", "dog_implied_prob_devigged",
    ])

    write_parquet(output, paths.FEATURE_STORE / "market_baseline.parquet")
    logger.info(f"Built market baseline: {len(output)} games")
    return output

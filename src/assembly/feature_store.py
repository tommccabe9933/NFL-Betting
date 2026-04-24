"""Final feature store assembly — one row per game, fav/dog perspective.

Joins all feature families into a single modeling-ready dataset.
Uses DuckDB for efficient SQL-based joins across parquet files.
"""

import logging

import polars as pl
import duckdb

from src import config, paths
from src.assembly.coverage_flags import has_pfr, has_ngs, has_ftn
from src.utils.io import write_parquet, read_parquet

logger = logging.getLogger(__name__)


def _join_team_features(
    games: pl.DataFrame,
    feature_df: pl.DataFrame,
    feature_prefix: str,
    join_keys: list = None,
) -> pl.DataFrame:
    """Join a team-level feature table for both fav and dog sides.

    For each game, joins features for fav_team and dog_team separately,
    then computes differentials (fav - dog).
    """
    if len(feature_df) == 0:
        return games

    if join_keys is None:
        join_keys = ["season", "week"]

    # Get feature columns (exclude join keys and team)
    feat_cols = [
        c for c in feature_df.columns
        if c not in join_keys + ["team", "game_id_norm"]
        and not c.startswith("_")
    ]

    if not feat_cols:
        return games

    # Ensure join key types match
    for key in join_keys + ["team"]:
        if key in feature_df.columns and key in games.columns:
            target_type = games[key].dtype
            if feature_df[key].dtype != target_type:
                feature_df = feature_df.with_columns(
                    pl.col(key).cast(target_type, strict=False)
                )

    # Prepare fav join: select only join keys + team + feature cols, then rename
    fav_src = feature_df.select(join_keys + ["team"] + feat_cols)
    fav_df = fav_src.rename({c: f"fav_{c}" for c in feat_cols})
    games = games.join(
        fav_df,
        left_on=join_keys + ["fav_team"],
        right_on=join_keys + ["team"],
        how="left",
        suffix="_fav_drop",
    )
    # Drop any stray right-side columns
    games = games.select([c for c in games.columns if not c.endswith("_fav_drop")])

    # Prepare dog join
    dog_src = feature_df.select(join_keys + ["team"] + feat_cols)
    dog_df = dog_src.rename({c: f"dog_{c}" for c in feat_cols})
    games = games.join(
        dog_df,
        left_on=join_keys + ["dog_team"],
        right_on=join_keys + ["team"],
        how="left",
        suffix="_dog_drop",
    )
    games = games.select([c for c in games.columns if not c.endswith("_dog_drop")])

    # Compute differentials for numeric columns
    for col in feat_cols:
        fav_col = f"fav_{col}"
        dog_col = f"dog_{col}"
        if (
            fav_col in games.columns
            and dog_col in games.columns
            and games[fav_col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ):
            games = games.with_columns(
                (pl.col(fav_col) - pl.col(dog_col)).alias(f"{col}_diff")
            )

    return games


def assemble_feature_store() -> pl.DataFrame:
    """Assemble the final feature store.

    Steps:
    1. Load market baseline as spine (has fav/dog assignment)
    2. Join each feature family for both fav and dog sides
    3. Compute target variable (fav_cover)
    4. Add coverage flags
    5. Filter to REG season, exclude pushes

    Output: data/processed/modeling/feature_store_v1.parquet
    """
    # 1. Load market baseline as spine
    market = read_parquet(paths.FEATURE_STORE / "market_baseline.parquet")

    # Load canonical games for scores (to compute target)
    games = read_parquet(paths.NORMALIZED / "canonical_games.parquet")
    games = games.filter(pl.col("game_type") == "REG")

    # Merge scores into market
    spine = market.join(
        games.select(["game_id_norm", "home_team", "away_team", "home_score", "away_score"]),
        on="game_id_norm",
        how="left",
    )

    # Compute actual margin and cover
    spine = spine.with_columns([
        # Fav margin: if fav is home, home_score - away_score; else away_score - home_score
        pl.when(pl.col("fav_at_home"))
        .then(pl.col("home_score") - pl.col("away_score"))
        .otherwise(pl.col("away_score") - pl.col("home_score"))
        .alias("fav_margin"),
    ])

    spine = spine.with_columns([
        # fav_cover: did the favorite cover the spread?
        # Margin + spread > 0 means cover (spread_fav is the magnitude, always positive)
        # R convention: spread_favorite is negative, so fav_margin > spread_favorite
        (pl.col("fav_margin") + pl.col("spread_fav")).alias("cover_margin"),
    ])

    spine = spine.with_columns([
        pl.when(pl.col("cover_margin") > 0).then(pl.lit("cover"))
        .when(pl.col("cover_margin") < 0).then(pl.lit("no_cover"))
        .otherwise(pl.lit("push"))
        .alias("fav_cover"),
    ])

    # Exclude pushes and games without scores
    spine = spine.filter(
        (pl.col("fav_cover") != "push")
        & pl.col("home_score").is_not_null()
    )

    # 2. Join feature families
    feature_files = {
        "epa_ratings": paths.FEATURE_STORE / "epa_ratings.parquet",
        "elo": paths.FEATURE_STORE / "elo.parquet",
        "rest_schedule": paths.FEATURE_STORE / "rest_schedule.parquet",
        "qb_features": paths.FEATURE_STORE / "qb_features.parquet",
        "injury_impact": paths.FEATURE_STORE / "injury_impact.parquet",
        "situational_epa": paths.FEATURE_STORE / "situational_epa.parquet",
        "tracking_proxy": paths.FEATURE_STORE / "tracking_proxy.parquet",
        "pfr_pressure": paths.FEATURE_STORE / "pfr_pressure.parquet",
        "ftn_scheme": paths.FEATURE_STORE / "ftn_scheme.parquet",
        "kaggle_team_stats": paths.FEATURE_STORE / "kaggle_team_stats.parquet",
    }

    result = spine

    for name, path in feature_files.items():
        if not path.exists():
            logger.warning(f"Feature file not found: {path}")
            continue

        feat_df = read_parquet(path)
        if len(feat_df) == 0:
            continue

        # Elo uses game_id_norm join, not season/week/team
        if name == "elo":
            # Join elo for fav
            fav_elo = feat_df.rename({"pre_elo": "fav_pre_elo"})
            result = result.join(
                fav_elo,
                left_on=["game_id_norm", "fav_team"],
                right_on=["game_id_norm", "team"],
                how="left",
            )
            # Join elo for dog
            dog_elo = feat_df.rename({"pre_elo": "dog_pre_elo"})
            result = result.join(
                dog_elo,
                left_on=["game_id_norm", "dog_team"],
                right_on=["game_id_norm", "team"],
                how="left",
            )
            # Elo diff
            if "fav_pre_elo" in result.columns and "dog_pre_elo" in result.columns:
                result = result.with_columns(
                    (pl.col("fav_pre_elo") - pl.col("dog_pre_elo")).alias("elo_diff")
                )
        else:
            result = _join_team_features(result, feat_df, name)

    # 3. Coverage flags
    result = result.with_columns([
        pl.col("season").map_elements(has_pfr, return_dtype=pl.Boolean).alias("has_pfr"),
        pl.col("season").map_elements(has_ngs, return_dtype=pl.Boolean).alias("has_ngs"),
        pl.col("season").map_elements(has_ftn, return_dtype=pl.Boolean).alias("has_ftn"),
    ])

    # 4. Bet time convention
    result = result.with_columns(pl.lit("closing").alias("bet_time_convention"))

    # 5. Join weather
    weather_path = paths.FEATURE_STORE / "weather.parquet"
    if weather_path.exists():
        weather = read_parquet(weather_path)
        result = result.join(
            weather.drop("season", "week"),
            on="game_id_norm",
            how="left",
        )

    write_parquet(result, paths.MODELING / "feature_store_v1.parquet")
    logger.info(
        f"Feature store assembled: {len(result)} games, "
        f"{len(result.columns)} columns"
    )
    return result

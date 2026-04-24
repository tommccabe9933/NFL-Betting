"""Weather features — temperature, wind bins, hinges."""

import logging

import polars as pl

from src import paths
from src.utils.io import write_parquet, read_parquet

logger = logging.getLogger(__name__)


def build_weather_features() -> pl.DataFrame:
    """Build weather features from canonical games.

    Output: one row per game_id_norm with:
    - temperature_f, wind_mph, humidity
    - wind bins (0-5, 6-10, 11-15, 16-20, 20+)
    - wind_over_12_hinge = max(0, wind - 12)
    - cold_game (temp < 32), outdoor_flag
    """
    games = read_parquet(paths.NORMALIZED / "canonical_games.parquet")
    games = games.filter(pl.col("game_type") == "REG")

    result = games.select([
        "game_id_norm", "season", "week",
        pl.col("weather_temperature").alias("temperature_f"),
        pl.col("weather_wind_mph").alias("wind_mph"),
        pl.col("weather_humidity").alias("humidity"),
        # Outdoor flag
        (~pl.col("roof").is_in(["dome", "closed"])).alias("outdoor_flag"),
    ]).with_columns([
        # Wind bins
        ((pl.col("wind_mph") >= 0) & (pl.col("wind_mph") <= 5)).alias("wind_bin_0_5"),
        ((pl.col("wind_mph") > 5) & (pl.col("wind_mph") <= 10)).alias("wind_bin_6_10"),
        ((pl.col("wind_mph") > 10) & (pl.col("wind_mph") <= 15)).alias("wind_bin_11_15"),
        ((pl.col("wind_mph") > 15) & (pl.col("wind_mph") <= 20)).alias("wind_bin_16_20"),
        (pl.col("wind_mph") > 20).alias("wind_bin_20_plus"),
        # Wind hinge at 12 mph
        pl.max_horizontal(pl.lit(0.0), pl.col("wind_mph") - 12).alias("wind_over_12_hinge"),
        # Cold game
        (pl.col("temperature_f") < 32).alias("cold_game"),
    ])

    write_parquet(result, paths.FEATURE_STORE / "weather.parquet")
    logger.info(f"Built weather features: {len(result)} games")
    return result

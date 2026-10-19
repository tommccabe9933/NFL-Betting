"""Build the canonical games table — one row per game_id.

Merges nflverse schedules with Spreadspoke market/weather data,
replicating the R pipeline's games_reg construction.
"""

import logging
from pathlib import Path

import polars as pl

from src.normalize.teams import (
    normalize_team_expr,
    normalize_game_id,
    TEAM_LOOKUP,
)
from src.utils.io import write_parquet, read_parquet_dir
from src import paths

logger = logging.getLogger(__name__)


def _load_schedules() -> pl.DataFrame:
    """Load all schedule parquet files from raw/schedules/."""
    return read_parquet_dir(paths.RAW_SCHEDULES)


def _load_spreadspoke() -> pl.DataFrame:
    """Load the Kaggle Spreadspoke games CSV and convert."""
    csv_path = paths.LEGACY_DATA / "nfl_all_seasons.csv"
    if not csv_path.exists():
        logger.warning(f"Spreadspoke CSV not found at {csv_path}")
        return pl.DataFrame()

    df = pl.read_csv(csv_path, infer_schema_length=10000, ignore_errors=True)
    # Normalize column names (janitor::clean_names equivalent)
    df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})
    # Cast weather columns to numeric (non-numeric like "rain" become null)
    for col in ["weather_temperature", "weather_wind_mph", "weather_humidity"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
    # Cast schedule_week to string (has "Conference", "Wildcard" etc for playoffs)
    if "schedule_week" in df.columns:
        df = df.with_columns(pl.col("schedule_week").cast(pl.Utf8))
    return df


def _resolve_team_name(name: str) -> str:
    """Resolve a full team name to a normalized abbreviation."""
    return TEAM_LOOKUP.get(name, name)


def build_canonical_games() -> pl.DataFrame:
    """Build the canonical games table.

    Returns one row per game with:
    - Identifiers: game_id, game_id_norm, season, week, game_type
    - Teams: home_team, away_team (normalized)
    - Scores: home_score, away_score
    - Market: spread_line, total_line, home_moneyline, away_moneyline
    - Context: div_game, roof, surface, stadium_neutral
    - Weather: weather_temperature, weather_wind_mph, weather_humidity
    - Derived: team_favorite_id, spread_favorite, fav_at_home
    """
    sched = _load_schedules()

    # Normalize team abbreviations in schedule data
    sched = sched.with_columns([
        normalize_team_expr("home_team"),
        normalize_team_expr("away_team"),
    ])

    # Build game_id_norm: {season}_{week:02d}_{away}_{home}
    sched = sched.with_columns(
        (
            pl.col("season").cast(pl.Utf8)
            + pl.lit("_")
            + pl.col("week").cast(pl.Utf8).str.zfill(2)
            + pl.lit("_")
            + pl.col("away_team")
            + pl.lit("_")
            + pl.col("home_team")
        ).alias("game_id_norm")
    )

    # Determine favorite from spread_line
    # In nflverse: positive spread_line means the home team is favored.
    # (Before v3.7 this was read backwards, swapping every fav_/dog_ column.)
    sched = sched.with_columns([
        pl.when(pl.col("spread_line") > 0)
        .then(pl.col("home_team"))
        .when(pl.col("spread_line") < 0)
        .then(pl.col("away_team"))
        .otherwise(pl.col("home_team"))  # pick'em defaults to home
        .alias("team_favorite_id"),

        pl.col("spread_line").abs().alias("spread_favorite"),
    ])

    # fav_at_home flag
    sched = sched.with_columns(
        (pl.col("team_favorite_id") == pl.col("home_team")).alias("fav_at_home")
    )

    # Merge weather from Spreadspoke parquet (already converted by market ingest)
    try:
        ss_path = paths.RAW_MARKET / "spreadspoke_games.parquet"
        if ss_path.exists():
            ss = pl.read_parquet(ss_path)
            # Spreadspoke has team names as full names — we need to match on season + week + home
            # Convert team names to abbreviations
            if "team_home" in ss.columns:
                ss = ss.with_columns([
                    pl.col("team_home").map_elements(
                        lambda x: TEAM_LOOKUP.get(x, x), return_dtype=pl.Utf8
                    ).alias("ss_home"),
                ])
                # Filter to matching seasons and extract weather
                weather_cols = [c for c in ["weather_temperature", "weather_wind_mph", "weather_humidity"]
                                if c in ss.columns]
                if weather_cols and "schedule_season" in ss.columns:
                    ss_weather = ss.select([
                        pl.col("schedule_season").cast(pl.Int64).alias("season"),
                        pl.col("schedule_week").cast(pl.Utf8).alias("week_str"),
                        pl.col("ss_home").alias("home_team"),
                    ] + [pl.col(c) for c in weather_cols])

                    # Join on season + home_team (week types may not match cleanly)
                    sched = sched.with_columns(pl.col("week").cast(pl.Utf8).alias("week_str"))
                    sched = sched.join(
                        ss_weather,
                        on=["season", "week_str", "home_team"],
                        how="left",
                    ).drop("week_str")
                    logger.info(f"Merged Spreadspoke weather: {sched['weather_temperature'].drop_nulls().len()} games with temp data")
    except Exception as e:
        logger.warning(f"Spreadspoke weather merge skipped: {e}")

    # Select final columns
    desired_cols = [
        "game_id", "game_id_norm", "season", "week", "game_type",
        "gameday", "gametime", "weekday",
        "home_team", "away_team", "home_score", "away_score",
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "team_favorite_id", "spread_favorite", "fav_at_home",
        "div_game", "roof", "surface",
        "weather_temperature", "weather_wind_mph", "weather_humidity",
        "stadium_neutral",
        "home_rest", "away_rest",
        "home_qb_name", "away_qb_name",
        "home_coach", "away_coach",
    ]
    # Only select columns that exist
    available = [c for c in desired_cols if c in sched.columns]
    result = sched.select(available)

    logger.info(
        f"Built canonical games table: {len(result)} games, "
        f"seasons {result['season'].min()}-{result['season'].max()}"
    )
    return result


def _merge_weather_safe(
    sched: pl.DataFrame,
    spreadspoke: pl.DataFrame,
) -> pl.DataFrame:
    """Safely merge weather from Spreadspoke, handling messy types."""
    try:
        return _merge_weather(sched, spreadspoke)
    except Exception as e:
        logger.warning(f"Weather merge failed: {e}")
        return sched


def _merge_weather(
    sched: pl.DataFrame,
    spreadspoke: pl.DataFrame,
) -> pl.DataFrame:
    """Merge weather data from Spreadspoke into schedule data.

    Uses season + week + team matching since game_id formats differ.
    """
    # Spreadspoke uses full team names — map to abbreviations
    if "team_home" in spreadspoke.columns:
        # Build a reverse lookup for Spreadspoke team names
        ss = spreadspoke.with_columns([
            pl.col("team_home").map_elements(
                lambda x: TEAM_LOOKUP.get(x, x), return_dtype=pl.Utf8
            ).alias("ss_home"),
            pl.col("team_away").map_elements(
                lambda x: TEAM_LOOKUP.get(x, x), return_dtype=pl.Utf8
            ).alias("ss_away"),
        ])

        # Extract weather columns
        weather_cols = []
        for col in ["weather_temperature", "weather_wind_mph", "weather_humidity"]:
            if col in ss.columns:
                weather_cols.append(col)

        if weather_cols and "schedule_season" in ss.columns:
            weather = ss.select(
                [
                    pl.col("schedule_season").alias("season"),
                    pl.col("schedule_week").cast(pl.Int64).alias("week"),
                    pl.col("ss_home").alias("home_team"),
                ]
                + [pl.col(c) for c in weather_cols]
            )

            # Add weather columns to sched where missing
            for col in weather_cols:
                if col not in sched.columns:
                    sched = sched.join(
                        weather.select("season", "week", "home_team", col),
                        on=["season", "week", "home_team"],
                        how="left",
                    )

    return sched


def build_and_save() -> pl.DataFrame:
    """Build canonical games and save to normalized directory."""
    games = build_canonical_games()
    write_parquet(games, paths.NORMALIZED / "canonical_games.parquet")
    return games

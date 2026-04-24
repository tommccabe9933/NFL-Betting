"""Rolling window utilities — faithful replica of R's lagged_roll().

CRITICAL: These functions are the leakage-safety foundation. The pattern is:
    1. Compute rolling mean over `window` games (including current)
    2. Lag by 1 game (shift forward)

This means: at game G, the feature value uses games G-window..G-1 (not G itself).

R implementation (scripts/00_project_setup.R:100-110):
    lagged_roll <- function(x, window) {
      lag(slide_dbl(x, ~mean(.x, na.rm=TRUE), .before=window-1, .after=0, .complete=FALSE))
    }
"""

from typing import List, Optional

import polars as pl


def lagged_rolling_mean(
    df: pl.DataFrame,
    value_col: str,
    window: int,
    group_col: str,
    sort_cols: List[str],
    alias: Optional[str] = None,
) -> pl.DataFrame:
    """Compute lagged rolling mean per group — exact replica of R's lagged_roll().

    For each group (team), sorted by sort_cols (season, week):
    1. rolling_mean(window) on value_col → uses current + (window-1) prior rows
    2. shift(1) → lag by 1 row

    Result: at game G, the value reflects games G-window through G-1.

    Args:
        df: Input DataFrame, must contain group_col, sort_cols, and value_col
        value_col: Column to compute rolling mean on
        window: Rolling window size (e.g. 5)
        group_col: Column to group by (e.g. "team")
        sort_cols: Columns to sort within group (e.g. ["season", "week"])
        alias: Output column name (default: "{value_col}_roll{window}")

    Returns:
        DataFrame with the rolling column added
    """
    if alias is None:
        alias = f"{value_col}_roll{window}"

    # Sort within groups then compute
    result = (
        df.sort(sort_cols)
        .with_columns(
            pl.col(value_col)
            .rolling_mean(window_size=window, min_samples=1)
            .shift(1)
            .over(group_col)
            .alias(alias)
        )
    )
    return result


def lagged_rolling_mean_multi(
    df: pl.DataFrame,
    value_cols: List[str],
    window: int,
    group_col: str,
    sort_cols: List[str],
    suffix: Optional[str] = None,
) -> pl.DataFrame:
    """Compute lagged rolling mean for multiple columns at once."""
    if suffix is None:
        suffix = f"_roll{window}"

    expressions = []
    for col in value_cols:
        expressions.append(
            pl.col(col)
            .rolling_mean(window_size=window, min_samples=1)
            .shift(1)
            .over(group_col)
            .alias(f"{col}{suffix}")
        )

    return df.sort(sort_cols).with_columns(expressions)


def season_cumulative_mean(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    season_col: str = "season",
    alias: Optional[str] = None,
) -> pl.DataFrame:
    """Compute expanding (season-to-date) mean, lagged by 1.

    At game G within a season, this is the mean of all prior games in that season.
    """
    if alias is None:
        alias = f"{value_col}_season_cum"

    return df.with_columns(
        pl.col(value_col)
        .cum_mean()
        .shift(1)
        .over([group_col, season_col])
        .alias(alias)
    )


def early_season_shrinkage(
    rolling_col: str,
    week_col: str = "week",
    prior_mean: float = 0.0,
    shrinkage_games: int = 5,
    alias: Optional[str] = None,
) -> pl.Expr:
    """Return a Polars expression that shrinks early-season rolling estimates toward prior.

    For weeks 1-shrinkage_games, blends the rolling estimate with the league mean
    based on how many games have been played. This addresses noisy early-season estimates.

    weight = min(week - 1, shrinkage_games) / shrinkage_games
    result = weight * rolling + (1 - weight) * prior_mean
    """
    if alias is None:
        alias = f"{rolling_col}_shrunk"

    weight = (
        pl.when(pl.col(week_col) <= shrinkage_games)
        .then((pl.col(week_col) - 1).cast(pl.Float64) / shrinkage_games)
        .otherwise(pl.lit(1.0))
    )

    return (
        weight * pl.col(rolling_col) + (1.0 - weight) * prior_mean
    ).alias(alias)

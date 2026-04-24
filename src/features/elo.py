"""Elo rating computation — faithful replica of R's compute_elo().

Sequential loop (cannot be parallelized). K=20, season regression toward 1500.
Reference: scripts/00_project_setup.R:114-159
"""

import logging

import polars as pl

from src import config, paths
from src.utils.io import write_parquet, read_parquet

logger = logging.getLogger(__name__)


def compute_elo(games: pl.DataFrame) -> pl.DataFrame:
    """Compute Elo ratings for all games.

    Args:
        games: DataFrame with columns: game_id_norm, season, week,
               home_team, away_team, home_score, away_score
               Must be sorted by (season, week).

    Returns:
        DataFrame with columns: game_id_norm, team, pre_elo
        Two rows per game (one per team).
    """
    # Sort chronologically
    games = games.sort(["season", "week"])

    teams = set(games["home_team"].to_list() + games["away_team"].to_list())
    elo = {t: config.ELO_INIT for t in teams}

    results = []
    prev_season = None

    for row in games.iter_rows(named=True):
        season = row["season"]
        game_id = row["game_id_norm"]
        home = row["home_team"]
        away = row["away_team"]
        score_h = row.get("home_score")
        score_a = row.get("away_score")

        # Season regression
        if prev_season is not None and season != prev_season:
            for t in elo:
                elo[t] = elo[t] * config.ELO_REGRESS + config.ELO_INIT * (1 - config.ELO_REGRESS)
        prev_season = season

        # Ensure teams exist in elo dict (new teams)
        for t in (home, away):
            if t not in elo:
                elo[t] = config.ELO_INIT

        # Record pre-game Elo
        results.append({"game_id_norm": game_id, "team": home, "pre_elo": elo[home]})
        results.append({"game_id_norm": game_id, "team": away, "pre_elo": elo[away]})

        # Update only if scores available
        if score_h is None or score_a is None:
            continue

        expected_home = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        actual_home = 1.0 if score_h > score_a else (0.0 if score_h < score_a else 0.5)

        elo[home] += config.ELO_K * (actual_home - expected_home)
        elo[away] += config.ELO_K * ((1 - actual_home) - (1 - expected_home))

    result_df = pl.DataFrame(results)
    logger.info(f"Computed Elo: {len(result_df)} team-game entries")
    return result_df


def build_elo_ratings() -> pl.DataFrame:
    """Build Elo ratings from canonical games and save."""
    games = read_parquet(paths.NORMALIZED / "canonical_games.parquet")
    elo_df = compute_elo(games)
    write_parquet(elo_df, paths.FEATURE_STORE / "elo.parquet")
    return elo_df

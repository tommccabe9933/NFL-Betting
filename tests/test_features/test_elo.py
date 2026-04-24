"""Tests for Elo rating computation."""

import pytest
import polars as pl
from src.features.elo import compute_elo


def test_elo_initial_rating():
    """All teams start at 1500."""
    games = pl.DataFrame({
        "game_id_norm": ["2023_01_KC_DET"],
        "season": [2023],
        "week": [1],
        "home_team": ["DET"],
        "away_team": ["KC"],
        "home_score": [21],
        "away_score": [20],
    })
    elo = compute_elo(games)
    assert len(elo) == 2
    assert all(e == pytest.approx(1500.0) for e in elo["pre_elo"].to_list())


def test_elo_winner_gains():
    """Winner's Elo should increase after a game."""
    games = pl.DataFrame({
        "game_id_norm": ["2023_01_KC_DET", "2023_02_DET_KC"],
        "season": [2023, 2023],
        "week": [1, 2],
        "home_team": ["DET", "KC"],
        "away_team": ["KC", "DET"],
        "home_score": [21, 14],
        "away_score": [20, 28],
    })
    elo = compute_elo(games)

    # After game 1: DET won, so DET's elo at game 2 > 1500
    det_game2 = elo.filter(
        (pl.col("game_id_norm") == "2023_02_DET_KC")
        & (pl.col("team") == "DET")
    )["pre_elo"][0]
    assert det_game2 > 1500.0


def test_elo_season_regression():
    """Elo should regress toward 1500 between seasons."""
    games = pl.DataFrame({
        "game_id_norm": ["2022_01_KC_DET", "2023_01_DET_KC"],
        "season": [2022, 2023],
        "week": [1, 1],
        "home_team": ["DET", "KC"],
        "away_team": ["KC", "DET"],
        "home_score": [35, None],
        "away_score": [10, None],
    })
    elo = compute_elo(games)

    # After 2022 game, DET has high Elo. After season regression, it should be closer to 1500
    det_2023 = elo.filter(
        (pl.col("game_id_norm") == "2023_01_DET_KC")
        & (pl.col("team") == "DET")
    )["pre_elo"][0]

    # Regressed = old * 0.75 + 1500 * 0.25
    # Should be between 1500 and the post-2022 value
    assert 1500 < det_2023 < 1600  # rough bounds


def test_elo_two_rows_per_game():
    """Each game should produce exactly 2 Elo rows."""
    games = pl.DataFrame({
        "game_id_norm": ["2023_01_A_B", "2023_02_C_D"],
        "season": [2023, 2023],
        "week": [1, 2],
        "home_team": ["B", "D"],
        "away_team": ["A", "C"],
        "home_score": [21, 14],
        "away_score": [20, 28],
    })
    elo = compute_elo(games)
    assert len(elo) == 4

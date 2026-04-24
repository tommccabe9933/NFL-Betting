"""Tests for rolling window utilities — CRITICAL for leakage safety.

Verifies that lagged_rolling_mean replicates R's lagged_roll() exactly:
  lag(slide_dbl(x, mean, .before=window-1, .after=0, .complete=FALSE))
"""

import pytest
import polars as pl
from src.features._rolling import lagged_rolling_mean


def _make_team_data(values: list, team: str = "KC") -> pl.DataFrame:
    """Helper to create a simple team-week DataFrame."""
    n = len(values)
    return pl.DataFrame({
        "season": [2023] * n,
        "week": list(range(1, n + 1)),
        "team": [team] * n,
        "metric": values,
    })


class TestLaggedRollingMean:
    """Verify rolling mean matches R's lagged_roll() behavior."""

    def test_basic_5game_window(self):
        """At game 6, should use mean of games 1-5 (lagged by 1)."""
        df = _make_team_data([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        result = lagged_rolling_mean(
            df, "metric", window=5,
            group_col="team", sort_cols=["season", "week"],
        )

        vals = result["metric_roll5"].to_list()

        # Week 1: lag of rolling_mean([10]) = None (no prior)
        assert vals[0] is None

        # Week 2: lag of rolling_mean([10]) shifted by 1 = 10.0
        assert vals[1] == pytest.approx(10.0)

        # Week 3: lag of rolling_mean([10,20]) = 15.0
        assert vals[2] == pytest.approx(15.0)

        # Week 4: lag of rolling_mean([10,20,30]) = 20.0
        assert vals[3] == pytest.approx(20.0)

        # Week 5: lag of rolling_mean([10,20,30,40]) = 25.0
        assert vals[4] == pytest.approx(25.0)

        # Week 6: lag of rolling_mean([10,20,30,40,50]) = 30.0
        assert vals[5] == pytest.approx(30.0)

        # Week 7: lag of rolling_mean([20,30,40,50,60]) = 40.0
        assert vals[6] == pytest.approx(40.0)

    def test_no_current_game_in_window(self):
        """The current game's value must NOT appear in its own rolling feature."""
        df = _make_team_data([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = lagged_rolling_mean(
            df, "metric", window=5,
            group_col="team", sort_cols=["season", "week"],
        )

        vals = result["metric_roll5"].to_list()

        # Week 1 value (100) should NOT affect week 1's rolling feature
        assert vals[0] is None  # no prior data

        # Week 2 should include week 1's 100
        assert vals[1] == pytest.approx(100.0)

    def test_multiple_teams_independent(self):
        """Rolling windows must be independent per team."""
        df1 = _make_team_data([10.0, 20.0, 30.0], team="KC")
        df2 = _make_team_data([100.0, 200.0, 300.0], team="SF")
        df = pl.concat([df1, df2])

        result = lagged_rolling_mean(
            df, "metric", window=5,
            group_col="team", sort_cols=["season", "week"],
        )

        kc = result.filter(pl.col("team") == "KC")["metric_roll5"].to_list()
        sf = result.filter(pl.col("team") == "SF")["metric_roll5"].to_list()

        # KC and SF should have independent rolling means
        assert kc[1] == pytest.approx(10.0)
        assert sf[1] == pytest.approx(100.0)

    def test_cross_season_continuity(self):
        """Rolling windows should span season boundaries (matches R behavior)."""
        df = pl.DataFrame({
            "season": [2022, 2022, 2022, 2023, 2023],
            "week": [16, 17, 18, 1, 2],
            "team": ["KC"] * 5,
            "metric": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        result = lagged_rolling_mean(
            df, "metric", window=3,
            group_col="team", sort_cols=["season", "week"],
        )

        vals = result["metric_roll3"].to_list()

        # 2023 week 1: should use 2022 weeks 16-18 data (lag of roll3 at row 3)
        # Roll3 at row 3 (week 18) = mean(10,20,30) = 20
        # Lagged to row 4 (2023 week 1) = 20
        assert vals[3] == pytest.approx(20.0)

    def test_window_3(self):
        """Test 3-game rolling window."""
        df = _make_team_data([10.0, 20.0, 30.0, 40.0, 50.0])
        result = lagged_rolling_mean(
            df, "metric", window=3,
            group_col="team", sort_cols=["season", "week"],
            alias="metric_roll3",
        )

        vals = result["metric_roll3"].to_list()
        assert vals[0] is None
        assert vals[1] == pytest.approx(10.0)
        assert vals[2] == pytest.approx(15.0)  # mean(10,20)
        assert vals[3] == pytest.approx(20.0)  # mean(10,20,30)
        assert vals[4] == pytest.approx(30.0)  # mean(20,30,40)

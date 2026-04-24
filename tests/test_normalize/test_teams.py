"""Tests for team normalization — verifies all 6 rules from R pipeline."""

import pytest
from src.normalize.teams import normalize_team, normalize_game_id, lookup_team_name


class TestNormalizeTeam:
    """Verify the 6-rule mapping from R's normalize_team()."""

    def test_lv_to_lvr(self):
        assert normalize_team("LV") == "LVR"

    def test_la_to_lar(self):
        assert normalize_team("LA") == "LAR"

    def test_oak_to_lvr(self):
        assert normalize_team("OAK") == "LVR"

    def test_sd_to_lac(self):
        assert normalize_team("SD") == "LAC"

    def test_stl_to_lar(self):
        assert normalize_team("STL") == "LAR"

    def test_jac_to_jax(self):
        assert normalize_team("JAC") == "JAX"

    def test_already_normalized_passthrough(self):
        assert normalize_team("KC") == "KC"
        assert normalize_team("NYG") == "NYG"
        assert normalize_team("SF") == "SF"

    def test_none_returns_none(self):
        assert normalize_team(None) is None


class TestNormalizeGameId:
    """Verify game_id normalization matches R's normalize_game_id()."""

    def test_sd_in_middle(self):
        assert normalize_game_id("2015_01_SD_KC") == "2015_01_LAC_KC"

    def test_sd_at_end(self):
        assert normalize_game_id("2015_01_KC_SD") == "2015_01_KC_LAC"

    def test_oak_to_lvr(self):
        assert normalize_game_id("2019_01_OAK_DEN") == "2019_01_LVR_DEN"

    def test_lv_to_lvr(self):
        assert normalize_game_id("2020_01_LV_CAR") == "2020_01_LVR_CAR"

    def test_stl_to_lar(self):
        assert normalize_game_id("2014_01_STL_MIN") == "2014_01_LAR_MIN"

    def test_la_to_lar(self):
        assert normalize_game_id("2016_01_LA_SF") == "2016_01_LAR_SF"

    def test_no_change_needed(self):
        assert normalize_game_id("2023_01_KC_DET") == "2023_01_KC_DET"


class TestTeamLookup:
    """Verify full name to abbreviation mapping."""

    def test_current_teams(self):
        assert lookup_team_name("Kansas City Chiefs") == "KC"
        assert lookup_team_name("San Francisco 49ers") == "SF"

    def test_historical_teams(self):
        assert lookup_team_name("Oakland Raiders") == "LVR"
        assert lookup_team_name("San Diego Chargers") == "LAC"
        assert lookup_team_name("St. Louis Rams") == "LAR"

    def test_washington_variants(self):
        assert lookup_team_name("Washington Redskins") == "WAS"
        assert lookup_team_name("Washington Football Team") == "WAS"
        assert lookup_team_name("Washington Commanders") == "WAS"

    def test_unknown_returns_none(self):
        assert lookup_team_name("Nonexistent Team") is None

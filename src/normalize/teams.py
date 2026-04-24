"""Team abbreviation normalization — faithful replica of R's normalize_team().

Handles franchise relocations and abbreviation inconsistencies across data sources.
"""

import re
from typing import Optional

import polars as pl


# Exact mapping from R's normalize_team() case_when (scripts/00_project_setup.R:88-98)
_TEAM_NORM_MAP = {
    "LV": "LVR",
    "LA": "LAR",
    "OAK": "LVR",
    "SD": "LAC",
    "STL": "LAR",
    "JAC": "JAX",
}

# Full name to abbreviation (from R's team_lookup tribble)
TEAM_LOOKUP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LVR",
    "Oakland Raiders": "LVR",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "St. Louis Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Redskins": "WAS",
    "Washington Football Team": "WAS",
    "Washington Commanders": "WAS",
}


def normalize_team(abbr: Optional[str]) -> Optional[str]:
    """Normalize a single team abbreviation. Returns None for None input."""
    if abbr is None:
        return None
    return _TEAM_NORM_MAP.get(abbr, abbr)


def normalize_team_expr(col: str) -> pl.Expr:
    """Return a Polars expression that normalizes a team abbreviation column."""
    expr = pl.col(col)
    for old, new in _TEAM_NORM_MAP.items():
        expr = pl.when(pl.col(col) == old).then(pl.lit(new)).otherwise(expr)
    return expr.alias(col)


def normalize_game_id(game_id: str) -> str:
    """Normalize a game_id string — replica of R's normalize_game_id().

    Replaces old team abbreviations in game_id format: {season}_{week}_{away}_{home}
    """
    replacements = [
        (r"_SD_", "_LAC_"), (r"_SD$", "_LAC"),
        (r"_OAK_", "_LVR_"), (r"_OAK$", "_LVR"),
        (r"_LV_", "_LVR_"), (r"_LV$", "_LVR"),
        (r"_STL_", "_LAR_"), (r"_STL$", "_LAR"),
        (r"_LA_", "_LAR_"), (r"_LA$", "_LAR"),
    ]
    result = game_id
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    return result


def normalize_game_id_expr(col: str) -> pl.Expr:
    """Return a Polars expression that normalizes game_id strings."""
    expr = pl.col(col)
    replacements = [
        ("_SD_", "_LAC_"), ("_SD$", "_LAC"),
        ("_OAK_", "_LVR_"), ("_OAK$", "_LVR"),
        ("_LV_", "_LVR_"), ("_LV$", "_LVR"),
        ("_STL_", "_LAR_"), ("_STL$", "_LAR"),
        ("_LA_", "_LAR_"), ("_LA$", "_LAR"),
    ]
    for pattern, replacement in replacements:
        # str.replace uses literal by default in polars
        expr = expr.str.replace(pattern, replacement)
    return expr.alias(col)


def lookup_team_name(full_name: str) -> Optional[str]:
    """Convert a full team name to normalized abbreviation."""
    return TEAM_LOOKUP.get(full_name)

"""Central configuration for the NFL pregame feature pipeline."""

MIN_SEASON = 2012
MAX_SEASON = 2025

ROLLING_WINDOW = 5
SHORT_ROLLING_WINDOW = 3

ELO_K = 20
ELO_REGRESS = 0.75
ELO_INIT = 1500

# Feature availability by season
FTN_ENABLED = True
FTN_MIN_SEASON = 2022
PFR_MIN_SEASON = 2018
NGS_MIN_SEASON = 2016
PARTICIPATION_MIN_SEASON = 2016

# Injury positional value weights (from R pipeline)
INJURY_POS_WEIGHTS = {
    "QB": 10, "LT": 5, "RT": 4, "LG": 3, "RG": 3, "C": 3,
    "WR": 4, "TE": 3, "RB": 3, "FB": 1,
    "CB": 5, "FS": 4, "SS": 4, "DE": 4, "DT": 3,
    "OLB": 3, "ILB": 3, "MLB": 3, "NT": 2,
    "K": 2, "P": 1, "LS": 1,
}

# Injury status severity weights
INJURY_STATUS_WEIGHTS = {
    "Out": 1.0,
    "Doubtful": 0.8,
    "Questionable": 0.3,
    "Probable": 0.05,
    "Doubtful - Loss": 0.8,
    "Out - Loss": 1.0,
}

# PBP columns to keep at ingest (drop ~380 unneeded columns)
PBP_KEEP_COLS = [
    "game_id", "play_id", "season", "week", "season_type",
    "posteam", "defteam", "down", "ydstogo", "yardline_100",
    "play_type", "pass", "rush", "epa", "cpoe", "air_yards",
    "yards_after_catch", "qb_hit", "sack", "interception",
    "fumble_lost", "touchdown", "first_down", "success",
    "shotgun", "no_huddle", "qb_dropback",
    "home_team", "away_team", "game_date",
    "passer_player_id", "rusher_player_id", "receiver_player_id",
    "passer_player_name", "rusher_player_name",
    "nflverse_game_id",
]

# nflverse direct download URL pattern
NFLVERSE_RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)

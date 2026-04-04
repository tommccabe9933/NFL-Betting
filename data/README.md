## Data

### Overview

This directory contains the raw datasets used for predicting NFL point spread coverage. The modeling dataset is built by `scripts/01_build_model_data.R` and saved to `objects/modeling_data.rds`.

### Data Sources

1. **NFL Team Data 2003-2023** (Nick Cantalupa, Kaggle)
   - `weekly_team_stats_offense.csv` -Weekly team-level offensive statistics
   - `weekly_team_stats_defense.csv` -Weekly team-level defensive statistics
   - Source: https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023

2. **NFL Scores and Betting Data** (Spreadspoke, Kaggle)
   - `nfl_all_seasons.csv` -Historical game results, closing spreads, totals, stadium info, weather
   - Source: https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data

3. **NFLverse / nflreadr** (live external data)
   - Play-by-play data used by `scripts/01_build_model_data.R` to compute EPA, success rate, and CPOE-derived features

### Codebook (Final Modeling Dataset)

The final modeling dataset (`objects/modeling_data.rds`) contains 3,538 game observations at the game level from the favorite's perspective, with the following 30 variables:

| Variable | Type | Description |
|----------|------|-------------|
| `game_id_norm` | Character | Normalized unique game identifier |
| `season` | Integer | NFL season year (2012–2024) |
| `week` | Integer | Regular season week number |
| `fav_cover` | Factor | **Target variable:** "cover" if the favorite covered the spread, "no_cover" otherwise |
| `spread_fav` | Numeric | Closing point spread for the favorite (negative values) |
| `over_under_line` | Numeric | Closing total points over/under line |
| `elo_diff` | Numeric | Elo rating differential (favorite minus underdog) |
| `ypp_diff` | Numeric | Yards per play differential (favorite minus underdog) |
| `pass_eff_diff` | Numeric | Pass efficiency differential |
| `rush_eff_diff` | Numeric | Rush efficiency differential |
| `pts_play_diff` | Numeric | Points per play differential |
| `third_down_diff` | Numeric | Third-down conversion rate differential |
| `turnover_diff` | Numeric | Turnover differential |
| `sack_rate_diff` | Numeric | Sack rate differential |
| `off_epa_diff` | Numeric | Offensive EPA (expected points added) differential |
| `def_epa_diff` | Numeric | Defensive EPA differential |
| `success_rate_diff` | Numeric | Play success rate differential |
| `momentum_diff` | Numeric | Momentum differential (rolling recent form vs season average) |
| `to_momentum_diff` | Numeric | Turnover momentum differential |
| `fav_roll5_ypp` | Numeric | Favorite's rolling 5-game yards per play |
| `fav_roll5_pts` | Numeric | Favorite's rolling 5-game points scored |
| `dog_roll5_ypp` | Numeric | Underdog's rolling 5-game yards per play |
| `dog_roll5_pts` | Numeric | Underdog's rolling 5-game points scored |
| `rest_diff` | Numeric | Rest differential in days (favorite minus underdog) |
| `fav_at_home` | Factor | Whether the favorite is the home team ("yes" / "no") |
| `fav_short_week` | Factor | Whether the favorite is on a short week ("yes" / "no") |
| `dog_short_week` | Factor | Whether the underdog is on a short week ("yes" / "no") |
| `stadium_neutral` | Logical | Whether the game is at a neutral site |
| `weather_temperature` | Numeric | Game-time temperature in degrees Fahrenheit |
| `weather_wind_mph` | Numeric | Wind speed in miles per hour |

### Notes

- Observation unit: one row per game, from the favorite's perspective
- Outcome classes are nearly balanced (~50/50 cover vs. no cover)
- Pushes (margin exactly equals the spread) are removed before modeling
- Time scope: 2012–2024 regular season games
- Weather data is missing for dome/indoor games; handled via median imputation in recipes

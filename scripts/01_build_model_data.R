# Build the modeling dataset from raw data sources

library(tidyverse)
library(tidymodels)
library(here)
library(janitor)
library(lubridate)
library(nflreadr)
library(slider)

source(here("scripts", "00_project_setup.R"))

# Load raw data
team_offense <- read_csv(here("data", "weekly_team_stats_offense.csv"), show_col_types = FALSE) |>
  clean_names()

team_defense <- read_csv(here("data", "weekly_team_stats_defense.csv"), show_col_types = FALSE) |>
  clean_names()

games_all <- read_csv(here("data", "nfl_all_seasons.csv"), show_col_types = FALSE) |>
  clean_names()

# Load play-by-play data for EPA features
pbp_seasons <- 2012:2023

pbp_raw <- map(pbp_seasons, ~ load_pbp(.x)) |>
  bind_rows()

# Compute EPA features from play-by-play
pbp_plays <- pbp_raw |>
  filter(
    season_type == "REG",
    rush == 1 | pass == 1,
    !is.na(epa), !is.na(posteam), posteam != ""
  )

epa_offense <- pbp_plays |>
  group_by(season, week, posteam) |>
  summarise(
    off_epa_play = mean(epa, na.rm = TRUE),
    off_pass_epa = mean(epa[pass == 1], na.rm = TRUE),
    off_rush_epa = mean(epa[pass == 0], na.rm = TRUE),
    off_success_rate = mean(epa > 0, na.rm = TRUE),
    off_cpoe = mean(cpoe, na.rm = TRUE),
    .groups = "drop"
  ) |>
  rename(team = posteam)

epa_defense <- pbp_plays |>
  group_by(season, week, defteam) |>
  summarise(
    def_epa_play = mean(epa, na.rm = TRUE),
    def_success_rate = mean(epa > 0, na.rm = TRUE),
    .groups = "drop"
  ) |>
  rename(team = defteam)

epa_weekly <- epa_offense |>
  left_join(epa_defense, by = c("season", "week", "team")) |>
  mutate(team = normalize_team(team))

# Process game schedule data
games_reg <- games_all |>
  filter(schedule_season >= 2012, schedule_playoff == FALSE) |>
  mutate(
    schedule_week = as.integer(schedule_week),
    game_date = mdy(schedule_date)
  ) |>
  left_join(team_lookup, by = c("team_home" = "team_name")) |>
  rename(home_team = team) |>
  left_join(team_lookup, by = c("team_away" = "team_name")) |>
  rename(away_team = team) |>
  mutate(
    team_favorite_id = normalize_team(team_favorite_id),
    game_id_norm = str_c(
      schedule_season, "_",
      str_pad(schedule_week, 2, pad = "0"), "_",
      away_team, "_", home_team
    ),
    team_favorite_id = if_else(is.na(team_favorite_id), home_team, team_favorite_id),
    spread_favorite = if_else(is.na(spread_favorite), 0, spread_favorite)
  )

# Combine offensive and defensive team stats
team_combined <- team_offense |>
  filter(season_type == "REG") |>
  inner_join(
    filter(team_defense, season_type == "REG"),
    by = c("game_id", "season", "week", "team", "season_type"),
    suffix = c("_off", "_def")
  ) |>
  mutate(
    game_id_norm = normalize_game_id(game_id),
    team_norm = normalize_team(team)
  )

# Handle potentially missing columns
expected_feature_columns <- c(
  "air_yards_off", "yards_after_catch_off", "receiving_yards_off",
  "shotgun_off", "fourth_down_converted_off", "fourth_down_failed_off",
  "qb_hit_def", "total_off_points_off"
)

missing_feature_columns <- setdiff(expected_feature_columns, names(team_combined))
team_combined[missing_feature_columns] <- NA_real_

# Engineer per-game efficiency features
team_features <- team_combined |>
  mutate(
    yards_per_play = safe_ratio(total_off_yards, offense_snaps),
    pass_eff = safe_ratio(passing_yards, pass_attempts),
    rush_eff = safe_ratio(rushing_yards, rush_attempts),
    comp_rate = safe_ratio(complete_pass, pass_attempts),
    third_down_rate = safe_ratio(third_down_converted, third_down_converted + third_down_failed),
    turnover_margin = (interception_def + fumble_forced_def) - (interception_off + fumble_lost_off),
    sack_rate = safe_ratio(sack, defense_snaps),
    pts_per_play = safe_ratio(total_off_points_off, offense_snaps),
    adot_off = safe_ratio(air_yards_off, pass_attempts),
    yac_share = safe_ratio(yards_after_catch_off, receiving_yards_off),
    shotgun_rate = safe_ratio(shotgun_off, offense_snaps),
    fourth_down_rate = safe_ratio(fourth_down_converted_off, fourth_down_converted_off + fourth_down_failed_off),
    qb_hit_rate = safe_ratio(qb_hit_def, defense_snaps)
  ) |>
  left_join(epa_weekly, by = c("season", "week", "team_norm" = "team"))

# Compute rolling averages with my various lags

team_rolling <- team_features |>
  group_by(team_norm) |>
  arrange(season, week, .by_group = TRUE) |>
  mutate(
    roll5_ypp = lagged_roll(yards_per_play, 5),
    roll5_pass_eff = lagged_roll(pass_eff, 5),
    roll5_rush_eff = lagged_roll(rush_eff, 5),
    roll5_turnover = lagged_roll(turnover_margin, 5),
    roll5_sack_rate = lagged_roll(sack_rate, 5),
    roll5_pts_play = lagged_roll(pts_per_play, 5),
    roll5_third_down = lagged_roll(third_down_rate, 5),
    roll5_off_epa = lagged_roll(off_epa_play, 5),
    roll5_def_epa = lagged_roll(def_epa_play, 5),
    roll5_success_rate = lagged_roll(off_success_rate, 5),
    roll3_ypp = lagged_roll(yards_per_play, 3),
    roll3_turnover = lagged_roll(turnover_margin, 3),
    roll3_pts_play = lagged_roll(pts_per_play, 3),
    season_baseline_ypp = lag(cummean(coalesce(yards_per_play, 0))),
    momentum_ypp = roll5_ypp - season_baseline_ypp
  ) |>
  ungroup()

# Compute Elo ratings
elo_input <- games_reg |>
  arrange(schedule_season, schedule_week) |>
  transmute(game_id_norm, home_team, away_team, season = schedule_season,
            week = schedule_week, score_home, score_away)

elo_log <- compute_elo(elo_input)

# Compute rest/schedule features
rest_data <- bind_rows(
  games_reg |> transmute(team = home_team, season = schedule_season, week = schedule_week, game_date),
  games_reg |> transmute(team = away_team, season = schedule_season, week = schedule_week, game_date)
) |>
  arrange(team, game_date) |>
  group_by(team) |>
  mutate(
    days_rest = as.numeric(game_date - lag(game_date)),
    days_rest = if_else(days_rest > 21, NA_real_, days_rest),
    short_week = coalesce(days_rest <= 6, FALSE)
  ) |>
  ungroup()

# Build game-level rows (one per team per game)
games_team <- games_reg |>
  transmute(
    game_id_norm, game_date, season = schedule_season, week = schedule_week,
    season_type = "REG", team_norm = home_team, opponent = away_team, home_away = "home",
    team_favorite_id, team_score = score_home, opponent_score = score_away,
    spread_favorite, over_under_line, stadium_neutral, weather_temperature, weather_wind_mph, weather_humidity
  ) |>
  bind_rows(
    games_reg |>
      transmute(
        game_id_norm, game_date, season = schedule_season, week = schedule_week,
        season_type = "REG", team_norm = away_team, opponent = home_team, home_away = "away",
        team_favorite_id, team_score = score_away, opponent_score = score_home,
        spread_favorite, over_under_line, stadium_neutral, weather_temperature, weather_wind_mph, weather_humidity
      )
  )

# join rolling features, Elo, and rest data
rolling_cols <- team_rolling |>
  select(game_id_norm, team_norm, season, week, starts_with("roll5_"), starts_with("roll3_"), momentum_ypp)

team_game_full <- games_team |>
  left_join(rolling_cols, by = c("game_id_norm", "team_norm", "season", "week")) |>
  left_join(rename(elo_log, team_elo = pre_elo), by = c("game_id_norm", "team_norm" = "team")) |>
  left_join(select(rest_data, team, season, week, days_rest, short_week), by = c("team_norm" = "team", "season", "week")) |>
  mutate(
    spread_team = if_else(team_norm == team_favorite_id, spread_favorite, -spread_favorite),
    margin = team_score - opponent_score,
    margin_vs_spread = margin + spread_team,
    cover = case_when(
      margin_vs_spread > 0 ~ "cover",
      margin_vs_spread < 0 ~ "no_cover",
      TRUE ~ "push"
    )
  )

# split into favorite and underdog perspectives
fav_rows <- team_game_full |>
  filter(team_norm == team_favorite_id, cover != "push") |>
  select(
    game_id_norm, season, week, fav_team = team_norm, fav_home = home_away,
    spread_fav = spread_favorite, over_under_line, weather_temperature, weather_wind_mph,
    stadium_neutral, fav_cover = cover, fav_elo = team_elo,
    fav_roll5_ypp = roll5_ypp, fav_roll5_pass = roll5_pass_eff, fav_roll5_rush = roll5_rush_eff,
    fav_roll5_to = roll5_turnover, fav_roll5_sack = roll5_sack_rate, fav_roll5_pts = roll5_pts_play,
    fav_roll5_3rd = roll5_third_down, fav_roll5_off_epa = roll5_off_epa, fav_roll5_def_epa = roll5_def_epa,
    fav_roll5_success = roll5_success_rate, fav_roll3_ypp = roll3_ypp, fav_roll3_to = roll3_turnover,
    fav_momentum_ypp = momentum_ypp, fav_days_rest = days_rest, fav_short_week = short_week
  )

# underdogs

dog_rows <- team_game_full |>
  filter(team_norm != team_favorite_id, cover != "push") |>
  select(
    game_id_norm, dog_team = team_norm, dog_elo = team_elo,
    dog_roll5_ypp = roll5_ypp, dog_roll5_pass = roll5_pass_eff, dog_roll5_rush = roll5_rush_eff,
    dog_roll5_to = roll5_turnover, dog_roll5_sack = roll5_sack_rate, dog_roll5_pts = roll5_pts_play,
    dog_roll5_3rd = roll5_third_down, dog_roll5_off_epa = roll5_off_epa, dog_roll5_def_epa = roll5_def_epa,
    dog_roll5_success = roll5_success_rate, dog_roll3_ypp = roll3_ypp, dog_roll3_to = roll3_turnover,
    dog_momentum_ypp = momentum_ypp, dog_days_rest = days_rest, dog_short_week = short_week
  )

# Build final modeling dataset with differential features
model_data <- fav_rows |>
  inner_join(dog_rows, by = "game_id_norm") |>
  mutate(
    elo_diff = fav_elo - dog_elo,
    ypp_diff = fav_roll5_ypp - dog_roll5_ypp,
    pass_eff_diff = fav_roll5_pass - dog_roll5_pass,
    rush_eff_diff = fav_roll5_rush - dog_roll5_rush,
    turnover_diff = fav_roll5_to - dog_roll5_to,
    sack_rate_diff = fav_roll5_sack - dog_roll5_sack,
    pts_play_diff = fav_roll5_pts - dog_roll5_pts,
    third_down_diff = fav_roll5_3rd - dog_roll5_3rd,
    off_epa_diff = fav_roll5_off_epa - dog_roll5_off_epa,
    def_epa_diff = fav_roll5_def_epa - dog_roll5_def_epa,
    success_rate_diff = fav_roll5_success - dog_roll5_success,
    momentum_diff = fav_roll3_ypp - dog_roll3_ypp,
    to_momentum_diff = fav_roll3_to - dog_roll3_to,
    rest_diff = coalesce(fav_days_rest, 7) - coalesce(dog_days_rest, 7),
    fav_at_home = factor(if_else(fav_home == "home", "yes", "no")),
    stadium_neutral = factor(stadium_neutral),
    fav_short_week = factor(if_else(coalesce(fav_short_week, FALSE), "yes", "no")),
    dog_short_week = factor(if_else(coalesce(dog_short_week, FALSE), "yes", "no")),
    fav_cover = factor(fav_cover, levels = c("cover", "no_cover"))
  ) |>
  select(
    game_id_norm, season, week, fav_cover,
    spread_fav, over_under_line, elo_diff,
    ypp_diff, pass_eff_diff, rush_eff_diff, pts_play_diff, third_down_diff,
    turnover_diff, sack_rate_diff, off_epa_diff, def_epa_diff, success_rate_diff,
    momentum_diff, to_momentum_diff,
    fav_roll5_ypp, fav_roll5_pts, dog_roll5_ypp, dog_roll5_pts,
    rest_diff, fav_at_home, fav_short_week, dog_short_week, stadium_neutral,
    weather_temperature, weather_wind_mph
  ) |>
  filter(!is.na(fav_cover), !is.na(spread_fav), !is.na(over_under_line))

# Summary statistics
data_summary <- tibble(
  measure = c(
    "Observation unit", "Outcome", "Seasons included",
    "Rows in final modeling data", "Predictor count",
    "Share favorite covers", "Share favorite does not cover"
  ),
  value = c(
    "One row per game from the favorite's perspective",
    "Whether the favorite covers the closing spread",
    paste(min(model_data$season), "to", max(model_data$season)),
    as.character(nrow(model_data)),
    as.character(ncol(model_data) - 4),
    scales::percent(mean(model_data$fav_cover == "cover")),
    scales::percent(mean(model_data$fav_cover == "no_cover"))
  )
)

write_csv(data_summary, here("results", "data_summary.csv"))
write_rds(model_data, here("objects", "modeling_data.rds"))
write_rds(team_game_full, here("objects", "team_game_full.rds"))
